package ml.animals;

import ml.titanic.TitanicUtils;
import java.util.Arrays;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.udf;

/**
 * Choose strategy to work with null data. Accuracy =  0.288
 */
public class Ex_3_Classified_with_SVM {

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
        Dataset<Row> vectorizedDF = assembler.transform(animals).select("features", "name", "type", "cyr_name", "Cyr_Class_Type");
        vectorizedDF.cache();

        // Step - 3: Train model
        LinearSVC classifier = new LinearSVC()
            .setMaxIter(100)
            .setRegParam(0.6)
            .setLabelCol("type");

        // Step - 4: Instantiate the One Vs Rest Classifier.
        OneVsRest multiClassTrainer = new OneVsRest().setClassifier(classifier).setLabelCol("type");

        // Step - 5: Train the multiclass model.
        OneVsRestModel model = multiClassTrainer.fit(vectorizedDF);

        // Step - 6: Print out all models
        Arrays.stream(model.models()).map(m -> (LinearSVCModel)m)
            .forEach(mdl -> System.out.println("Coefficients " + mdl.coefficients().toString() + " intercept " + mdl.intercept()));

        Dataset<Row> rawPredictions = model.transform(vectorizedDF);

        Dataset<Row> predictions = enrichPredictions(spark, rawPredictions);

        predictions.show(100);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("type")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }



    private static Dataset<Row> enrichPredictions(SparkSession spark, Dataset<Row> rawPredictions) {
        Dataset<Row> classNames = spark.read()
            .option("inferSchema", "true")
            .option("charset", "windows-1251")
            .option("header", "true")
            .csv("/home/zaleslaw/data/cyr_class.csv");

        Dataset<Row> prClassNames = classNames.select(classNames.col("Class_Number").as("pr_class_id"), classNames.col("Cyr_Class_Type").as("pr_class_type"));
        Dataset<Row> enrichedPredictions = rawPredictions.join(prClassNames, rawPredictions.col("prediction").equalTo(prClassNames.col("pr_class_id")));

        rawPredictions.show();

        UserDefinedFunction checkClasses = udf(
            (String type, String prediction) -> type.equals(prediction) ? "" : "ERROR", DataTypes.StringType
        );

        Dataset<Row> dataset = enrichedPredictions
            .withColumn("Error", checkClasses.apply(enrichedPredictions.col("Cyr_Class_Type"), enrichedPredictions.col("pr_class_type")));
        Dataset<Row> predictions = dataset
            .select(
                dataset.col("name"),
                dataset.col("cyr_name").as("Name"),
                dataset.col("Cyr_Class_Type").as("Real_class_type"),
                dataset.col("pr_class_type").as("Predicted_class_type"))
            .orderBy(dataset.col("Error").desc());

        return predictions;
    }

    private static Dataset<Row> readAnimalDataset(SparkSession spark) {
        Dataset<Row> animals = spark.read()
            .option("inferSchema", "true")
            .option("charset", "windows-1251")
            .option("header", "true")
            .csv("/home/zaleslaw/data/cyr_animals.csv");

        animals.show();

        Dataset<Row> classNames = spark.read()
            .option("inferSchema", "true")
            .option("charset", "windows-1251")
            .option("header", "true")
            .csv("/home/zaleslaw/data/cyr_class.csv");

        classNames.show(false);

        Dataset<Row> animalsWithClassTypeNames = animals.join(classNames, animals.col("type").equalTo(classNames.col("Class_Number")));

        return animalsWithClassTypeNames;
    }
}
