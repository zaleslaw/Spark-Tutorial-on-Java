package rdd;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;

import static rdd.util.SparkUtil.getSparkSession;

/**
 * Make dataset based on range and extract RDD from it
 */
public class Ex_0_RDD_vs_Dataset {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("RDD_vs_Dataset");

        Dataset ds = spark.range(100000000);
        System.out.println(ds.count());
        System.out.println(ds.toJavaRDD().count());
    }
}
