import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{udf, _}

object Classification {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-sql-basic")
      .master("local[*]")
      .getOrCreate()

    val train_csv = "C:/Users/itim/IdeaProjects/Francisco/data/train.csv"
    val test_csv = "C:/Users/itim/IdeaProjects/Francisco/data/test.csv"

    // Load and parse the data file, converting it to a DataFrame.
    val crimes = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv(train_csv);

    val dayOrNight = udf {
      (h: Int) =>
        if (h > 5 && h < 18) {
          "Day"
        } else {
          "Night"
        }
    }

    val weekend = udf {
      (day: String) =>
        if (day == "Sunday" || day == "Saturday") {
          "Weekend"
        } else {
          "NotWeekend"
        }
    }

    val df = crimes
      .withColumn("HourOfDay", hour(col("Dates")))
      .withColumn("Month", month(col("Dates")))
      .withColumn("Year", year(col("Dates")))
      .withColumn("HourOfDay", hour(col("Dates")))

    val df1 = df
      .withColumn("DayOrNight", dayOrNight(col("HourOfDay")))
      .withColumn("Weekend", weekend(col("DayOfWeek")))

    val featureCols = Array("X", "Y")
    val assembler1 = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("XandY")

    val uberFeatures = assembler1.transform(df1)

    //Traing KMeans model
    val kmeans = new KMeans()
      .setK(10)
      .setFeaturesCol("XandY")
      .setPredictionCol("Clusters")
      .setMaxIter(20)

    val model1 = kmeans.fit(uberFeatures)

    println("Final Centers: ")
    model1.clusterCenters.foreach(println)

    val predictions1 = model1.transform(uberFeatures)
    predictions1.show

    var categoryIndex = new StringIndexer().setInputCol("Category").setOutputCol("CategoryIndex")
    var dayIndex = new StringIndexer().setInputCol("DayOfWeek").setOutputCol("DayOfWeekIndex")
    var districtIndex = new StringIndexer().setInputCol("PdDistrict").setOutputCol("PdDistrictIndex")
    // var addressIndex = new StringIndexer().setInputCol("Address").setOutputCol("AddressIndex")
    var dayNightIndex = new StringIndexer().setInputCol("DayOrNight").setOutputCol("DayOrNightsIndex")

    val assembler = new VectorAssembler().setInputCols(Array(
      "DayOfWeekIndex", "PdDistrictIndex", "HourOfDay", "Month", "Clusters"))
      .setOutputCol("indexedFeatures")

    val Array(training, test) = predictions1.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier()
      .setLabelCol("CategoryIndex")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(10)
      .setMaxBins(100)

    val pipeline = new Pipeline()
      .setStages(Array(categoryIndex, dayIndex, districtIndex, assembler, rf))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("CategoryIndex")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    /*

        val paramGrid = new ParamGridBuilder()
          .addGrid(rf.impurity, Array("entropy", "gini"))
          .build()
    */

    /*val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
*/
    val model = pipeline.fit(training)

    val predictions = model.transform(test)
    predictions.show()
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
  }
}

