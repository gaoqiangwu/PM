# file: train.py
# author: BINWong
# time: 2019/1/4 08:53
# Copyright 2019 BINWong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import os
import tensorflow as tf
from poems.model import rnn_model
from poems.poems import process_poems, generate_batch

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', os.path.abspath('./model'), 'model save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poems.txt'), '诗歌档案名称.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS


def run_training():
    # 检测模型参数文件夹及父文件夹, 不存在则新建
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    # 读取诗集文件
    # 依次得到数字ID表示的诗句、汉字-ID的映射map、所有的汉字的列表
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)

    # 按照batch读取输入和输出数据
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    # 声明输入、输出的占位符
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    # 通过rnn模型得到结果状态集
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    # 初始化saver和session
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0

        # 加载上次的模型参数(如果有)
        checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))

            # 如果有模型参数, 则取出对应的epoch, 训练从该epoch开始训练
            start_epoch += int(checkpoint.split('-')[-1])

        # 开始训练
        print('## start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0

                # 计算一个epoch需要多少次batch训练完, 有余数则忽略掉末尾部分
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    # 训练并计算loss
                    # batches_inputs[n]: 第n个batch的输入数据
                    # batches_outputs[n]: 第n个batch的输出数据
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch, batch, loss))

                # 每训练6个epoch进行一次模型保存
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            # 用户手动退出时, 尝试保存模型参数
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()