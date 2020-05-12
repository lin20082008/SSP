# -*- coding:utf-8 -*-

# python dis_tf_ssp.py --job_name=ps --task_index=0
# python dis_tf_ssp.py --job_name=worker --task_index=0
# python dis_tf_ssp.py --job_name=worker --task_index=1

import time
import numpy as np
import tensorflow as tf

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import state_ops, variables, variable_scope
from tensorflow.python.training import session_run_hook

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "172.172.0.10:10000",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "172.172.0.11:10001,172.172.0.12:10002,172.172.0.13:10003",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


# @tf_export("train.SemiSyncRunHook")
# class SemiSyncRunHook(session_run_hook.SessionRunHook):
class SemiSyncRunHook(tf.train.SessionRunHook):
    """Run by SSP."""

    def __init__(self, index, worker_count, staleness=10):
        """Initializes a `SemiSyncRunHook`.
        Args:
          index: work index
          worker_count: number of workers
          staleness:
        """

        if index >= worker_count:
            print("worker index {} is bigger than worker_count {}".format(index, worker_count))
            return

        self._const_max_test_step = 10000
        self._last_step = 0  # 上一次 wait 的步骤数
        self._last_time = self._now_time()  # 上一次 wait 的时间

        self._index = index
        self._staleness = staleness
        self._wait_time = 0.01  # 等待时间，单位：秒；这个时间不能设置的太长，跟 worker 的训练速度和 staleness 相关
        self._worker_steps = []  # 记录 worker 训练步骤数的变量列表

        for i in range(worker_count):
            worker_step = variable_scope.variable(0, trainable=False, name="worker_step_" + str(i))
            self._worker_steps.append(worker_step)
            if i == index:
                self._my_step_update_op = state_ops.assign_add(worker_step, 1)

        self._initialize_op = variables.variables_initializer(self._worker_steps)

    def _now_time(self):
        return time.time()

    def after_create_session(self, session, coord):
        session.run(self._initialize_op)  # 初始化记录 worker 训练步骤数的变量

    def before_run(self, run_context):
        run_context.session.run(self._my_step_update_op)  # 更新本 worker 的训练步骤数
        return None

    def after_run(self, run_context, run_values):
        while True:
            # 1. 获取所有 worker 的训练步骤数
            all_worker_steps = run_context.session.run(self._worker_steps)
            # print("all worker steps={}. my work id={}".format(all_worker_steps, self._index))

            # 2. 如果训练当前 worker 的训练步骤数 > 最小 worker 训练步骤数 + staleness，sleep(10ms); 否则 break;
            if all_worker_steps[self._index] > min(all_worker_steps) + self._staleness:
                diff_step = all_worker_steps[self._index] - self._last_step
                if diff_step / self._const_max_test_step > 1:
                    self._wait_time = (self._now_time() - self._last_time) / diff_step * self._staleness * 0.7

                    # 更新
                    self._last_step = all_worker_steps[self._index]
                    self._last_time = self._now_time()

                time.sleep(self._wait_time)  # 等待慢 worker 执行
                # print("all worker steps={}, my work id={}. waiting {}s...".format(all_worker_steps, self._index, self._wait_time))
            else:
                break


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    worker_count = len(worker_hosts)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            X = tf.placeholder(tf.float32)
            Y = tf.placeholder(tf.float32)
            w = tf.Variable(0.0, name="weight")
            b = tf.Variable(0.0, name="reminder")
            y = w * X + b

            loss = tf.reduce_mean(tf.square(y - Y))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            # 更新梯度
            train_op = optimizer.minimize(loss, global_step=global_step)

            hooks = [tf.train.StopAtStepHook(last_step=1000000)]

            semiSyncRunHook = SemiSyncRunHook(FLAGS.task_index, worker_count=worker_count, staleness=10)
            hooks.append(semiSyncRunHook)

            with tf.train.MonitoredTrainingSession(
                    master=server.target, is_chief=(FLAGS.task_index == 0),
                    checkpoint_dir="./ssp_saved_model",
                    hooks=hooks) as mon_sess:
                while not mon_sess.should_stop():
                    train_x = np.random.randn(1)
                    train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step = mon_sess.run([train_op, loss, global_step], feed_dict={X: train_x, Y: train_y})
                    if step % steps_to_validate == 0:
                        w_, b_ = mon_sess.run([w, b])
                        print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w_, b_, loss_v))


if __name__ == "__main__":
    tf.app.run()
