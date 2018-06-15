from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf

NO_OF_STEPS =20000
NORMALISER = 1000
PATH= ".\\Concrete.csv" #Path of the csv file
TRAIN_FRACTION=0.7
DEPENDENT_VARIABLE_NAME="compressivestrength"

#get the train sample set
def in_training_set(line):
    #Split Data
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(TRAIN_FRACTION * num_buckets)

#get the test sample
def in_test_set(line):
    return ~in_training_set(line)

#decode the csv
def parse_csv(line):
    example_defaults = collections.OrderedDict([
        ("Cement", [0.0]),
        ("BlastFurnaceSlag", [0.0]),
        ("FlyAsh", [0.0]),
        ("Water", [0.0]),
        ("Superplasticizer", [0.0]),
        ("CoarseAggregate", [0.0]),
        ("Fine.Aggregate", [0.0]),
        ("Age", [0]),
        ("compressivestrength", [0.0])])

    items = tf.decode_csv(line, list(example_defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(example_defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(DEPENDENT_VARIABLE_NAME)

    return features_dict, label


def main(argv):
  """Builds, trains, and evaluates the model."""
  assert len(argv) == 1

  # Normalize the labels to units of thousands for better convergence.
  def normalize_price(features, labels):
      return features, labels / NORMALISER


  dataset = tf.data.TextLineDataset(PATH)

  train_dataset = (dataset
                   # Take only the training-set lines.
                   .filter(in_training_set)
                   # Cache data so you only read the file once.
                   .cache()
                   # Decode each line into a (features_dict, label) pair.
                   .map(parse_csv).map(normalize_price))

  test_dataset = (dataset.filter(in_test_set).cache().map(parse_csv).map(normalize_price))

  feature_columns = [
      tf.feature_column.numeric_column(key="Cement"),
      tf.feature_column.numeric_column(key="BlastFurnaceSlag"),
      tf.feature_column.numeric_column(key="FlyAsh"),
      tf.feature_column.numeric_column(key="Water"),
      tf.feature_column.numeric_column(key="Superplasticizer"),
      tf.feature_column.numeric_column(key="CoarseAggregate"),
      tf.feature_column.numeric_column(key="Fine.Aggregate"),
      tf.feature_column.numeric_column(key="Age")

  ]


  #DNNRegressor, with 20,20 hiddenlayers
  model = tf.estimator.DNNRegressor(
      hidden_units=[20, 20], feature_columns=feature_columns)

  # train input function
  def input_train():
      return (
          # Shuffling with a buffer,larger than the dataset
          train_dataset.shuffle(1000).batch(128)
              # Repeat forever
              .repeat().make_one_shot_iterator().get_next())

  # test_input_fn.
  def input_test():
      return (test_dataset.shuffle(1000).batch(128)
              .make_one_shot_iterator().get_next())



  # Training
  model.train(input_fn=input_train, steps=NO_OF_STEPS)

  # Evaluate
  eval_result = model.evaluate(input_fn=input_test)

  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: {:.0f}"
        .format(NORMALISER * average_loss ** 0.5))

  print()

  #Average_loss= 0.151 #Error 3%



if __name__ == "__main__":
  # INFO -> Loss, Step, time(sec) for each step
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
