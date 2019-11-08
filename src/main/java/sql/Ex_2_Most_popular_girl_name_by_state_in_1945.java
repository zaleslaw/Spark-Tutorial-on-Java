package sql;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.max;
import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * Task - Get most popular girl name in each state in 1945
 */
public class Ex_2_Most_popular_girl_name_by_state_in_1945 {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Most_popular_girl_name_by_state_in_1945");

        // ASSERT: Files should exists
        Dataset<Row> stateNames = spark.read().parquet(DATA_DIRECTORY + "/stateNames");
        stateNames.show();
        stateNames.printSchema();

        Dataset<Row> filteredStateNames = stateNames
            .where("Year=1945 and Gender='F'")
            .select("Name", "State", "Count");

        filteredStateNames.cache();

        filteredStateNames.orderBy(col("state").desc(), col("Count").desc()).show();

        Dataset<Row> stateAndCount = filteredStateNames
            .groupBy("state")
            .agg(max("Count").as("max"));

        stateAndCount.show();

        // Self-join, of course
        Dataset<Row> stateAndName = filteredStateNames
            .join(stateAndCount,
                stateAndCount.col("max").equalTo(filteredStateNames.col("Count")).and(
                    stateAndCount.col("state").equalTo(filteredStateNames.col("state")))
            )
            .select(filteredStateNames.col("state"), col("Name").alias("name")) // should choose only String names or $Columns
            .orderBy(col("state").desc(), col("Count").desc());

        stateAndName.printSchema();
        stateAndName.show(100);
        stateAndName.explain();
        stateAndName.explain(true);
    }
}
