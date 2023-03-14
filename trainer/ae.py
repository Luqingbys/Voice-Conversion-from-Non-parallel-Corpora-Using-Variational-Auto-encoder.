import logging
import os

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from util.image import make_png_jet_thumbnail, make_png_thumbnail
from trainer.gan import GANTrainer


class VAETrainer(GANTrainer):
    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters.
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        optimizer = tf.train.AdamOptimizer(lr, b1, b2)

        g_vars = tf.trainable_variables()

        with tf.name_scope('Update'):
            opt_g = optimizer.minimize(self.loss['G'], var_list=g_vars, global_step=global_step)
        return {
            'g': opt_g,
            'global_step': global_step
        }


    def _refresh_status(self, sess):
        fetches = {
            # "Toal MSE": self.loss['Toal MSE'],
            "Avg MSE": self.loss['Avg loss'],
            "step": self.opt['global_step'],
        }
        # fetches = {
        #     "D_KL": self.loss['D_KL'],
        #     "logP": self.loss['logP'],
        #     "step": self.opt['global_step'],
        # }
        result = sess.run(
            fetches=fetches,
            # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            # run_metadata=run_metadata,
        )

        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open(os.path.join(dirs['logdir'], 'timeline.ctf.json'), 'w') as fp:
        #     fp.write(trace.generate_chrome_trace_format())

        # Message
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'Avg loss = {:.3e} '.format(result['Avg loss'])
        # msg += 'Total MSE = {:.3e} '.format(result['Total'])
        print('\r{}'.format(msg), end='', flush=True)
        logging.info(msg)


    # def _validate(self, machine, n=10):
    #     N = n * n

    #     # same row same z
    #     z = tf.random_normal(shape=[n, self.arch['z_dim']])
    #     z = tf.tile(z, [1, n])
    #     z = tf.reshape(z, [N, -1])
    #     z = tf.Variable(z, trainable=False, dtype=tf.float32)

    #     # same column same y
    #     y = tf.range(0, 10, 1, dtype=tf.int64)
    #     y = tf.reshape(y, [-1,])
    #     y = tf.tile(y, [n,])

    #     Xh = machine.generate(z, y) # 100, 64, 64, 3
    #     Xh = make_png_thumbnail(Xh, n)
    #     return Xh


    def train(self, nIter, machine=None, summary_op=None):
        # Xh = self._validate(machine=machine, n=10)

        run_metadata = tf.RunMetadata()

        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            # summary_writer=summary_writer,
            # summary_op=None,
            # is_chief=True,
            save_model_secs=300,
            global_step=self.opt['global_step'])


        # sess_config = configure_gpu_settings(args.gpu_cfg)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        with sv.managed_session(config=sess_config) as sess:
            sv.loop(60, self._refresh_status, (sess,))
            for step in range(self.arch['training']['max_iter']):
                if sv.should_stop():
                    break

                # main loop
                sess.run(self.opt['g'])

                # # output img
                # if step % 1000 == 0:
                #     xh = sess.run(Xh)
                #     with tf.gfile.GFile(
                #         os.path.join(
                #             self.dirs['logdir'],
                #             'img-anime-{:03d}k.png'.format(step // 1000),
                #         ),
                #         mode='wb',
                #     ) as fp:
                #         fp.write(xh)