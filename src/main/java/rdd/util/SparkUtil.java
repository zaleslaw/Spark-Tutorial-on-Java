package rdd.util;

import org.apache.spark.sql.SparkSession;

public class SparkUtil {
    public static final String DATA_DIRECTORY = "/home/zaleslaw/data";

    public static SparkSession getSparkSession(String appName) {
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
            .master("local[2]")
            .appName(appName)
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }

    public static void println(String message) {
        System.out.println(message);
    }
}
