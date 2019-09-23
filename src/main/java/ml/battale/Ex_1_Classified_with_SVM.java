package ml.battale;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;

/**
 * I added one new column "eatable" and marked all observations in the dataset cyr_binarized_animals
 *
 * You can see that SVM doesn't work well.
 * The possible explanation: both classes are not linear separable.
 */
public class Ex_1_Classified_with_SVM {

    public static void main(String[] args) {
        SparkSession spark = getSparkSession();

        // Step - 0: Overview the animal dataset and read it
        Dataset<Row> animals = readAnimalDataset(spark);

        animals.show();
        animals.printSchema();
        Dataset<Row> persisted = animals.persist(StorageLevel.MEMORY_ONLY());
        long initTime = System.currentTimeMillis();
        System.out.println("Loaded at " + initTime);

        // Step - 1: Make Vectors from dataframe's columns using special VectorAssembler object
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[] {"_c1", "_c2"})
            .setOutputCol("features");


        // Step - 2: Transform dataframe to vectorized dataframe
        Dataset<Row> vectorizedDF = assembler.transform(persisted).select("features", "_c0");


        // Step - 3: Train model
        /*LinearSVC classifier = new LinearSVC()
            .setMaxIter(100)
            .setRegParam(0.3)
            .setLabelCol("_c0");

        LinearSVCModel mdl = classifier.fit(vectorizedDF);*/

        DecisionTreeClassifier classifier = new DecisionTreeClassifier().setLabelCol("_c0");
        DecisionTreeClassificationModel mdl = classifier.fit(vectorizedDF);

        //System.out.println("Coefficients " + mdl.coefficients().toString() + " intercept " + mdl.intercept());
        System.out.println("Training time: " + (System.currentTimeMillis() - initTime));

        Dataset<Row> rawPredictions = mdl.transform(vectorizedDF);


        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
            .setLabelCol("_c0")
            .setRawPredictionCol("prediction");

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

    }


    private static Dataset<Row> readAnimalDataset(SparkSession spark) {
        Dataset<Row> animals = spark.read()
            .option("inferSchema", "true")
            .option("charset", "windows-1251")
            .option("delimiter", ";")
            .csv("D:\\dataset_medium.txt");

        return animals;
    }

    private static SparkSession getSparkSession(){
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
            .master("local[2]")
            .config("spark.executor.memory", "16g")
            .config("spark.driver.memory",  "8g")
            .appName("Spark_SQL")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }
}
