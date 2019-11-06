package rdd;

import java.util.Arrays;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.StatCounter;

import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Set theory operations on the RDDs.
 */
public class Ex_7_Statistics {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Set_theory");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        JavaDoubleRDD anomalInts = jsc.parallelizeDoubles(Arrays.asList(1.0, 1.0, 2.0, 2.0, 3.0, 150.0, 1.0, 2.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, -100.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0), 3);
        StatCounter stats = anomalInts.stats();
        double stddev = stats.stdev();
        double mean = stats.mean();
        println("Stddev is " + stddev + " mean is " + mean);

        JavaDoubleRDD normalInts = anomalInts.filter(x -> (Math.abs(x - mean) < 3 * stddev));
        normalInts.collect().forEach(System.out::println);
    }
}
