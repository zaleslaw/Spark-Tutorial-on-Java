package sql;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.udf;
import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Demonstrates the Spark SQL API.
 */
public class Ex_5_Spark_SQL_UDF {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Spark_SQL_UDF");

        Dataset<Row> stateNames = spark.read().parquet(DATA_DIRECTORY + "/stateNames");
        stateNames.show();
        stateNames.printSchema();

        stateNames.cache();

        // Step-1: Define and register UDF function
        UDF1<Integer, Boolean> lambdaIsWorldWarTwoYear = year -> year >= 1939 && year <= 1945;
        spark.sqlContext().udf().register("isWorldWarTwoYear", lambdaIsWorldWarTwoYear, DataTypes.BooleanType);

        // Step-2.1: Use UDF in dataframe (old way)
        println("Old way UDF result");
        stateNames
            .select(col("Year"), callUDF("isWorldWarTwoYear", col("Year")))
            .distinct()
            .orderBy(col("Year").desc())
            .show(100);

        // Step-2.2: Use UDF in dataframe (using selectExpr)
        println("SelectExpr UDF result");
        stateNames
            .selectExpr("Year", "isWorldWarTwoYear(Year)")
            .distinct()
            .orderBy(col("Year").desc())
            .show(150);

        // Step-2.3: Use UDF in dataframe (Scala way)
        println("Scala way UDF result");

        UserDefinedFunction lambdaIsWorldWarOneYear = udf(
            (Integer year) -> year >= 1914 && year <= 1918, DataTypes.BooleanType
        );

        stateNames
            .select(col("Year"), lambdaIsWorldWarOneYear.apply(col("Year")))
            .distinct()
            .orderBy(col("Year").desc())
            .show(150);

        // Step-3: Use the same UDF in SQL expression
        stateNames.createOrReplaceTempView("stateNames");

        // Step-4: Get full list of boy names who was born during WWII
        spark.sql("SELECT DISTINCT Name FROM stateNames WHERE Gender = 'M' and isWorldWarTwoYear(Year) ORDER BY Name DESC").show(150);
    }
}
