package internals;

import java.util.List;
import java.util.stream.IntStream;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.storage.StorageLevel;

import static java.util.stream.Collectors.toList;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Shows how to print out information about partitions and change the shuffling streams of data via Custom Partitioner.
 */
public class Ex_3_RDD_Internals {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("RDD_Internals");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        List<Integer> data = IntStream.range(0, 10_000_000).boxed().collect(toList());
        // Make RDD
        JavaRDD<Integer> intRDD = jsc.parallelize(data, 1024);
        long countRDD = intRDD.persist(StorageLevel.MEMORY_ONLY()).count();
        println(String.valueOf(countRDD));

        JavaRDD<Row> rowRdd = intRDD.map(RowFactory::create);
        rowRdd.toDebugString();

        StructField[] fields = new StructField[] {
            DataTypes.createStructField("value", IntegerType, true)};

        StructType schema = DataTypes.createStructType(fields);

        Dataset<Row> dataframe = new SQLContext(jsc).createDataFrame(rowRdd, schema);

        long countDF = dataframe.persist(StorageLevel.MEMORY_ONLY()).count();
        println(String.valueOf(countDF));

        while (true) {

        }
    }
}
