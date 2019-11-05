package rdd;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import static rdd.SparkUtil.DATA_DIRECTORY;

/**
 * Creates the RDD and saves to file.
 */
public class Ex_1_RDD_To_File {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession();

        // Makes RDD based on Array
        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        List<Integer> r = Arrays.asList(1, 2, 3, 4, 5);

        // Creates RDD with 3 parts
        JavaRDD<Integer> ints = jsc.parallelize(r, 3);

        ints.saveAsTextFile(DATA_DIRECTORY + "/ints"); // works for windows well
    }

    private static SparkSession getSparkSession() {
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
            .master("local[2]")
            .appName("RDD_Intro")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }
}
