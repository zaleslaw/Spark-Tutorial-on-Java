package ml;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class IrisSample {

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

    private static Dataset<Row> readIrisData(SparkSession spark){
        Dataset<Row> irisData = spark.read()
                .text("src/main/resources/iris.txt");

        irisData.printSchema();



        return irisData;
    }

    public static void main(String[] args) {
        SparkSession spark = getSparkSession();

        Dataset<Row> irisRawData = readIrisData(spark);

        irisRawData.show();
    }
}
