package ml;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TitanicSample {

    private static SparkSession getSparkSession(){
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
                .master("local")
                .appName("Spark_SQL")
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }

    private static Dataset<Row> readPassengers(SparkSession spark){
        Dataset<Row> passengers = spark.read()
                .option("delimiter", ";")
                .option("inferSchema", "true")
                .option("header", "true")
                .csv("/home/zaleslaw/data/titanic.csv");

        passengers.printSchema();

        passengers.show();

        return passengers;
    }

    public static void main(String[] args) {
        SparkSession spark = getSparkSession();

        Dataset<Row> passengers = readPassengers(spark);

        // Step - 1: Make Vectors from dataframe's columns using special Vector Assmebler
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"pclass", "sibsp", "parch"})
                .setOutputCol("features");

        // Step - 2: Transform dataframe to vectorized dataframe with dropping rows
        Dataset<Row> output = assembler.transform(
                passengers.na().drop(new String[]{"pclass", "sibsp", "parch"}) // <============== drop row if it has nulls/NaNs in the next list of columns
        ).select("features", "survived");

        // Step - 3: Set up the Decision Tree Classifier
        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        // Step - 4: Train the model
        DecisionTreeClassificationModel model = trainer.fit(output);

        // Step - 5: Predict with the model
        Dataset<Row> rawPredictions = model.transform(output);

        // Step - 6: Evaluate prediction
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Step - 7: Calculate accuracy
        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        // Step - 8: Print out the model
        System.out.println("Learned classification tree model:\n" + model.toDebugString());
    }
}
