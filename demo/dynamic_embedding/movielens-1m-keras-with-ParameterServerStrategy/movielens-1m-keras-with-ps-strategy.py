import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import os
from tensorflow_recommenders_addons import dynamic_embedding as de

parser = argparse.ArgumentParser()
parser.add_argument("--worker",
                    type=str,
                    help="like: localhost:2231,localhost:2232")
parser.add_argument("--ps",
                    type=str,
                    help="like: localhost:2221,localhost:2222")
parser.add_argument("--task_type", type=str, help="must in [ps, worker]")
parser.add_argument("--task_index", type=int, help="like: 0")

arguments = parser.parse_args()


class DualChannelsDeepModel(tf.keras.Model):

  def init(self,
           strategy,
           user_embedding_size=1,
           movie_embedding_size=1,
           embedding_initializer=None):
    super(DualChannelsDeepModel, self).init()
    self.user_embedding_size = user_embedding_size
    self.movie_embedding_size = movie_embedding_size

    if embedding_initializer is None:
      embedding_initializer = tf.keras.initializers.Zeros()

    self.user_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size=user_embedding_size,
        initializer=embedding_initializer,
        name='user_embedding',
        distribute_strategy=strategy)
    self.movie_embedding = de.keras.layers.SquashedEmbedding(
        embedding_size=movie_embedding_size,
        initializer=embedding_initializer,
        name='movie_embedding',
        distribute_strategy=strategy)

    self.dnn1 = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.dnn2 = tf.keras.layers.Dense(
        16,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.dnn3 = tf.keras.layers.Dense(
        5,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.bias_net = tf.keras.layers.Dense(
        5,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

  def call(self, features):
    user_id = tf.reshape(features['user_id'], (-1, 1))
    movie_id = tf.reshape(features['movie_id'], (-1, 1))
    user_latent = self.user_embedding(user_id)
    movie_latent = self.movie_embedding(movie_id)
    latent = tf.concat([user_latent, movie_latent], axis=1)

    x = self.dnn1(latent)
    x = self.dnn2(x)
    x = self.dnn3(x)

    bias = self.bias_net(latent)
    x = 0.2 * x + 0.8 * bias
    return x


def get_dataset(batch_size=1):
  dataset = tfds.load('movielens/1m-ratings', split='train')
  features = dataset.map(
      lambda x: {
          "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
          "user_id": tf.strings.to_number(x["user_id"], tf.int64),
      })
  ratings = dataset.map(
      lambda x: tf.one_hot(tf.cast(x['user_rating'] - 1, dtype=tf.int64), 5))
  dataset = dataset.zip((features, ratings))
  dataset = dataset.shuffle(4096, reshuffle_each_iteration=False)
  if batch_size > 1:
    dataset = dataset.batch(batch_size)
  return dataset


def create_strategy(args):
  ps_hosts = args.ps.split(",")
  worker_hosts = args.worker.split(",")
  env = {
      "cluster": {
          "ps": ps_hosts,
          "worker": worker_hosts
      },
      "task": {
          "type": args.task_type,
          "index": args.task_index
      }
  }

  import json
  os.environ['TF_CONFIG'] = json.dumps(env)
  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  os.environ["GRPC_FAIL_FAST"] = "use_caller"
  server = tf.distribute.Server(cluster_resolver.cluster_spec(),
                                job_name=cluster_resolver.task_type,
                                task_index=cluster_resolver.task_id,
                                protocol=cluster_resolver.rpc_layer or "grpc",
                                start=True)
  if cluster_resolver.task_type == "ps":
    print("parameter server starting...")
    server.join()
  else:
    print("worker training starting...")
    variable_partitioner = (
        tf.distribute.experimental.partitioners.MinSizePartitioner(
            min_shard_bytes=256 << 10, max_shards=len(ps_hosts)))
    return tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver, variable_partitioner=variable_partitioner)


strategy = create_strategy(args=arguments)

embedding_size = 32
epochs = 10
steps_per_epoch = 20000
model_dir = "./model"
test_batch = 1024
test_steps = 128

with strategy.scope():
  model = DualChannelsDeepModel(strategy, embedding_size, embedding_size,
                                tf.keras.initializers.RandomNormal(0.0, 0.5))
  optimizer = tf.keras.optimizers.Adam(1E-3)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer)

  auc = tf.keras.metrics.AUC(num_thresholds=1000)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[
                    auc,
                ])
  dataset = get_dataset(batch_size=32)
  model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2)
