import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from tensorflow.contrib import slim
import tf_slim as slim
from util.image import nchw_to_nhwc
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu)


class AE(object):
    def __init__(self, arch, is_training=False):
        '''
        Variational auto-encoder implemented in 2D convolutional neural nets
        使用二维卷积神经网络的变分自编码器
        Input:
            `arch`: dict， network architecture (`dict`)，见 architecture-vae-vcc2016.json
            `is_training`: (unused now) it was kept for historical reasons (for `BatchNorm`)
        '''
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'], # 10
                self.arch['z_dim'], # 128
                'y_embedding')

        self._generate = tf.make_template(
            'Generator',
            self._generator)

        self._encode = tf.make_template(
            'Encoder',
            self._encoder)

        self.generate = self.decode  # for VAE-GAN extension


    def _sanity_check(self):
        for net in ['encoder', 'generator']:
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel']) == len(self.arch[net]['stride'])


    def _unit_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim])
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings


    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        return slim.bias_add(x)


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        ''' 
        n_class: 10，表示类别数量，比如十个人，所有的语音属于十个人
        h_dim: 128，表示编码器的编码结果（隐变量）的维度，此处是128
        scope_name: 'y_embedding'
        '''
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        # embeddings: (n_class, h_dim), (10, 128)
        return embeddings


    def _encoder(self, x, is_training=None):
        net = self.arch['encoder']
        ''' 
        encoder: {
        "kernel": [[7, 1], [7, 1], [7, 1], [7, 1], [7, 1]],
		"stride": [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1]],
		"output": [16, 32, 64, 128, 256],
		"l2-reg": 1e-6
        }
        '''
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv2d-{}'.format(i)
            )
        x = slim.flatten(x)
        # tf.layers.dense(inputs, units)，inputs是输入数据，units是输出维度大小（改变inputs最后一维）
        z_mu = tf.layers.dense(x, self.arch['z_dim']) # z_dim: 128
        z_lv = tf.layers.dense(x, self.arch['z_dim']) # z_dim: 128
        return z_mu, z_lv


    def _generator(self, z, y, is_training=None):
        ''' 生成器，也就是接收隐状态z和外来唯一标识y，重构输出（学习原始输入） '''
        net = self.arch['generator']
        ''' 
        net = {
        "hwc": [19, 1, 81],
		"merge_dim": 171,
		"kernel": [[9, 1], [7, 1], [7, 1], [1025, 1]],
		"stride": [[3, 1], [3, 1], [3, 1], [1, 1]],
		"output": [32, 16, 8, 1],
		"l2-reg": 1e-6
        }
        '''
        h, w, c = net['hwc']

        if y is not None:
            y = tf.nn.embedding_lookup(self.y_emb, y)
            x = self._merge([z, y], h * w * c)
        else:
            x = z

        x = tf.reshape(x, [-1, c, h, w])  # channel first

        # 四个逆置卷积
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = tf.layers.conv2d_transpose(x, o, k, s,
                padding='same',
                data_format='channels_first',
            )
            if i < len(net['output']) -1:
                # 第1、2、3均在逆置卷积层后加上批标准化和激活
                x = Layernorm(x, [1, 2, 3], 'ConvT-LN{}'.format(i))
                x = lrelu(x)
        return x


    def loss(self, x, y):
        ''' 
        x为原始输入（特征提取之后的）,(16, 1, 513, 1)，实际上是提取到的语音特征，(n, c, h, w)
        y为额外添加的语者唯一标识，(16,)，实际上是语者（总共10个语者）      
        '''
        with tf.name_scope('loss'):
            # z_mu, z_lv = self._encode(x) # z_mu、z_lv均为张量
            # z = GaussianSampleLayer(z_mu, z_lv)
            # xh = self._generate(z, y)

            # D_KL = tf.reduce_mean(
            #     GaussianKLD(
            #         slim.flatten(z_mu),
            #         slim.flatten(z_lv),
            #         slim.flatten(tf.zeros_like(z_mu)),
            #         slim.flatten(tf.zeros_like(z_lv)),
            #     )
            # )
            # logPx = tf.reduce_mean(
            #     GaussianLogDensity(
            #         slim.flatten(x),
            #         slim.flatten(xh),
            #         tf.zeros_like(slim.flatten(xh))),
            # )
            z, _ = self._encode(x) # z: (16, 128)
            xh = self._generate(z, y) # xh: (16, 1, 513, 1), x: (16, 1, 513, 1)
            # mse = tf.losses.mean_squared_error(slim.flatten(x), slim.flatten(xh))
            l = tf.sqrt(tf.reduce_mean((xh - x)**2))


        loss = dict()
        # loss的三个部分
        # loss['G'] = - logPx + D_KL
        # loss['D_KL'] = D_KL
        # loss['logP'] = logPx
        # loss['Toal loss'] = mse
        loss['Avg loss'] = l

        # tf.summary.scalar('KL-div', D_KL)
        # tf.summary.scalar('logPx', logPx)
        # tf.summary.scalar('Total MSE', mse)
        tf.summary.scalar('Avg loss', l)

        tf.summary.histogram('xh', xh)
        tf.summary.histogram('x', x)
        return loss


    def encode(self, x):
        ''' 编码器 '''
        z_mu, _ = self._encode(x)
        return z_mu


    def decode(self, z, y):
        ''' 解码器 '''
        xh = self._generate(z, y)
        return nchw_to_nhwc(xh)