package sql;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.MutableAggregationBuffer;
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.StringType;
import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Use UDAF to find the longest names for each year in the given dataset.
 */
public class Ex_6_Spark_SQL_UDAF {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Spark_SQL_UDAF ");

        Dataset<Row> stateNames = spark.read().parquet(DATA_DIRECTORY + "/stateNames");
        stateNames.show();
        stateNames.printSchema();

        stateNames.cache();

        // Step-1: Use the same UDF in SQL expression
        stateNames.createOrReplaceTempView("stateNames");

        // Step-2: Get names with max length by year
        spark.sql("SELECT Year, MAX(Name) FROM stateNames GROUP BY Year ORDER BY Year").show(100); // <= this approach doesn't work as we wish due to lexicographical order

        // Step-3: Register UDAF function
        spark.udf().register("LONGEST_WORD", new LongestWord());

        // Step-4: Get pairs <Year, Name with max length>
        spark.sql("SELECT Year, LONGEST_WORD(Name) FROM stateNames GROUP BY Year ORDER BY Year").show(100); // <= this approach doesn't work as we wish due to lexicographical order

    }

    public static class LongestWord extends UserDefinedAggregateFunction {

        @Override public StructType inputSchema() {
            return new StructType().add("name", StringType, true);
        }

        @Override public StructType bufferSchema() {
            return new StructType().add("maxLengthWord", StringType, true);
        }

        @Override public org.apache.spark.sql.types.DataType dataType() {
            return StringType;
        }

        @Override public boolean deterministic() {
            return true;
        }

        @Override public void initialize(MutableAggregationBuffer buffer) {
            println(">>> initialize (buffer: " + buffer.toString() + ")");
            // NOTE: Scala's update used under the covers
            buffer.update(0, "");
        }

        @Override public void update(MutableAggregationBuffer buffer, Row input) {
            println(">>> compare (buffer: " + buffer.toString() + " -> input: " + input.toString() + ")");
            String maxWord = buffer.getString(0);
            String currentName = input.getString(0);
            if (currentName.length() > maxWord.length())
                buffer.update(0, currentName);
        }

        @Override public void merge(MutableAggregationBuffer buffer, Row row) {
            println(">>> merge (buffer: " + buffer.toString() + " -> row: " + row.toString() + ")");
            String maxWord = buffer.getString(0);
            String currentName = row.getString(0);
            if (currentName.length() > maxWord.length())
                buffer.update(0, currentName);
        }

        @Override public Object evaluate(Row buffer) {
            println(">>> evaluate (buffer: " + buffer.toString() + ")");

            return buffer.getString(0);
        }
    }
}
