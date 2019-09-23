package ml.eatable;

import ml.titanic.TitanicUtils;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.udf;

/**
 * Dataset marked from the Data Scientist's point ov view.
 *
 * But the DT works well for this case.
 */
public class Ex_2_Classified_with_DecisionTree {

    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        // Step - 0: Overview the animal dataset and read it
        Dataset<Row> animals = readAnimalDataset(spark);

        animals.show();

        // Step - 1: Make Vectors from dataframe's columns using special VectorAssembler object
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[] {"hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"})
            .setOutputCol("features");


        // Step - 2: Transform dataframe to vectorized dataframe
        Dataset<Row> vectorizedDF = assembler.transform(animals).select("features", "eatable", "cyr_name");
        vectorizedDF.cache();

        // Step - 3: Train model
        DecisionTreeClassifier classifier = new DecisionTreeClassifier()
            .setLabelCol("eatable");

        DecisionTreeClassificationModel mdl = classifier.fit(vectorizedDF);

        System.out.println(mdl.toDebugString());

        Dataset<Row> rawPredictions = mdl.transform(vectorizedDF);

        Dataset<Row> predictions = enrichPredictions(spark, rawPredictions);

        predictions.show(100, false);

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
            .setLabelCol("eatable")
            .setRawPredictionCol("prediction");

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }



    private static Dataset<Row> enrichPredictions(SparkSession spark, Dataset<Row> rawPredictions) {
        UserDefinedFunction checkClasses = udf(
            (Integer type, Double prediction) -> type.intValue() == prediction.intValue() ? "" : "ERROR", DataTypes.StringType
        );

        Dataset<Row> dataset = rawPredictions
            .withColumn("Error", checkClasses.apply(rawPredictions.col("eatable"), rawPredictions.col("prediction")));
        Dataset<Row> predictions = dataset.select(
            dataset.col("cyr_name").as("Name"),
            dataset.col("eatable"),
            dataset.col("prediction"))
            .orderBy(dataset.col("Error").desc());

        return predictions;
    }

    private static Dataset<Row> readAnimalDataset(SparkSession spark) {
        Dataset<Row> animals = spark.read()
            .option("inferSchema", "true")
            .option("charset", "windows-1251")
            .option("header", "true")
            .csv("/home/zaleslaw/data/cyr_binarized_animals.csv");

        return animals;
    }
}
