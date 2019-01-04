# file: compose_poem.py
# author: BINWong
# time: 2019/1/4 10:40
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


import tensorflow as tf
from poems.model import rnn_model
from poems.poems import process_poems
import numpy as np

start_token = 'B'
end_token = 'E'
model_dir = './model/'
corpus_file = './data/poems.txt'

lr = 0.0002


def to_word(predict, vocabs):
    # 取词逻辑
    # 将predict累加求和
    # 求出预测可能性的总和
    predict = predict[0]       
    predict /= np.sum(predict)

    # 返回将0~s的随机值插值到t中的索引值
    # 由于predict各维度对应的词向量是按照训练数据集的频率进行排序的
    # 故P(x|predict[i]均等时) > P(x + δ), 即达到了权衡优先取前者和高概率词向量的目的
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_poem(begin_word):
    # 根据首个汉字作诗
    # 作诗时, batch_size设为1
    batch_size = 1
    print('## loading corpus from %s' % model_dir)

    # 读取诗集文件
    # 依次得到数字ID表示的诗句、汉字-ID的映射map、所有的汉字的列表
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)
    # 声明输入的占位符
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    # 通过rnn模型得到结果状态集
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)
    # 初始化saver和session
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # 加载上次的模型参数
        checkpoint = tf.train.latest_checkpoint(model_dir)

        # 注: 无模型参数时, 该步直接crash, 强制有训练好的模型参数
        saver.restore(sess, checkpoint)
        # 取出诗文前缀(G)对应的索引值所谓初始输入
        x = np.array([list(map(word_int_map.get, start_token))])
        # 得出预测值和rnn的当前状态
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            # 用户输入值赋值给word
            word = begin_word
        else:
            # 若未输入, 则取初始预测值的词向量
            word = to_word(predict, vocabularies)

        # 初始化作诗结果变量
        poem_ = ''

        i = 0
        # 未到结束符时, 一直预测下一个词
        while word != end_token:
            # 没预测一个则追加到结果上
            poem_ += word
            i += 1
            if i >= 24:
                break
            # 初始化输入为[[0]]
            x = np.zeros((1, 1))

            # 赋值为当前word对应的索引值
            x[0, 0] = word_int_map[word]

            # 根据当前词和当前的上下文状态(last_state)进行预测
            # 返回的结果是预测值和最新的上下文状态
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})

            # 根据预测值得出词向量
            word = to_word(predict, vocabularies)

        return poem_


def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


if __name__ == '__main__':
    begin_char = input('##请输入第一个字符##:')
    poem = gen_poem(begin_char)
    pretty_print_poem(poem_=poem)