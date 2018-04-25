#! /usr/bin/env python3
# coding:utf-8
import json

import tornado.web
import tornado.ioloop
import engine_for_cnn as engine
import tensorflow as tf


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        text = self.get_argument('text')
        print(text)
        predict = self.classify(text)
        data = {
            'text': text,
            'predict': predict[0]
        }
        self.write(json.dumps({'data': data}).encode('utf-8'))

    def classify(self, text):
        sample = engine.text_tensor(text, engine.wv)
        tensor_proto = tf.contrib.util.make_tensor_proto(sample, shape=[1, len(sample[0]), 200])
        engine.request.inputs['x'].CopyFrom(tensor_proto)
        response = engine.stub.Predict(engine.request, 10.0)
        result = list(response.outputs['y'].int64_val)
        return result


def make_app():
    return tornado.web.Application([
        (r"/predict", MainHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(909)
    print('listen start')
    tornado.ioloop.IOLoop.current().start()
