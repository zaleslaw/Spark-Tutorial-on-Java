package sql;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * Demonstrates the Spark SQL API.
 */
public class Ex_3_Spark_SQL {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Spark_SQL");

        // ASSERT: Files should exists
        Dataset<Row> stateNames = spark.read().parquet(DATA_DIRECTORY + "/stateNames");
        stateNames.show();
        stateNames.printSchema();

        stateNames.createOrReplaceTempView("stateNames");

        // Step-1: Get full list of boy names
        spark.sql("SELECT DISTINCT Name FROM stateNames WHERE Gender = 'M' ORDER BY Name").show(100);

        // Step-2: Get proportion of state NY births in total births
        Dataset<Row> nationalNames = spark.read().json("/home/zaleslaw/data/nationalNames");

        nationalNames.createOrReplaceTempView("nationalNames");

        Dataset<Row> result = spark.sql("SELECT nyYear as year, stateBirths/usBirths as proportion, stateBirths, usBirths FROM (SELECT year as nyYear, SUM(count) as stateBirths FROM stateNames WHERE state = 'NY' GROUP BY year ORDER BY year) as NY" +
            " JOIN (SELECT year as usYear, SUM(count) as usBirths FROM nationalNames GROUP BY year ORDER BY year) as US ON nyYear = usYear");

        result.show(150);
        result.explain(true);
    }
}
