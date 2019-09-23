package ml.animals;

import ml.titanic.TitanicUtils;
import java.util.Arrays;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.collect_list;

/**
 * Choose strategy to work with null data. Accuracy =  0.288
 */
public class Ex_1_Clustering_with_KMeans {

    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        // Step - 0: Overview the animal dataset and read it
        Dataset<Row> animals = readAnimalDataset(spark);

        animals.show();

        // Step - 1: Make Vectors from dataframe's columns using special VectorAssembler object
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[] {"hair", "milk", "eggs"}) // + add "eggs"
            // .setInputCols(new String[] {"hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"})
            .setOutputCol("features");


        // Step - 2: Transform dataframe to vectorized dataframe
        Dataset<Row> vectorizedDF = assembler.transform(animals).select("features", "cyr_name", "Cyr_Class_Type");
        vectorizedDF.cache();

        for (int i = 2; i <= 20; i++) {

            System.out.println("Clusterize with " + i + " clusters");

            // Step - 3: Train model
            KMeans kMeansTrainer = new KMeans()
                .setK(i) // possible number of clusters: change it from 2 to 10, optimize parameter
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("cluster");

            KMeansModel model = kMeansTrainer.fit(vectorizedDF);

            // Step - 4: Print out the sum of squared distances of points to their nearest center
            double SSE = model.computeCost(vectorizedDF);
            System.out.println("Sum of Squared Errors = " + SSE);

            // Step - 5: Print out cluster centers
            System.out.println("Cluster Centers: ");
            Arrays.stream(model.clusterCenters()).forEach(System.out::println);

            System.out.println("Real clusters and predicted clusters");
            Dataset<Row> predictions = model.summary().predictions();

            // Step - 6: Print out predicted and real classes

            System.out.println("Predicted classes");

            predictions
                .select("cyr_name", "cluster")
                .groupBy("cluster")
                .agg(collect_list("cyr_name"))
                .orderBy("cluster")
                .show((int)predictions.count(), false);

        }

        System.out.println("Real classes");
        vectorizedDF
            .select("cyr_name", "Cyr_Class_Type")
            .groupBy("Cyr_Class_Type")
            .agg(collect_list("cyr_name")).show((int)vectorizedDF.count(), false);
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
