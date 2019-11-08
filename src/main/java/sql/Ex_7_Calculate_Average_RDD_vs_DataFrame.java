package sql;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.apache.spark.sql.types.DataTypes.StringType;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * Demonstrates the Spark SQL API.
 */
public class Ex_7_Calculate_Average_RDD_vs_DataFrame {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Spark_SQL");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        List<String> salariesData = Arrays.asList(
            "John 1900 January",
            "Mary 2000 January",
            "John 1800 February",
            "John 1000 March",
            "Mary 1500 February",
            "Mary 2900 March"
        );

        // Creates RDD with 3 parts
        JavaRDD<String[]> salaries = jsc.parallelize(salariesData, 3).map(s -> s.split(" "));

        Function2<Tuple2<Double, Integer>, Tuple2<Double, Integer>, Tuple2<Double, Integer>> reduceFunc
            = (t1, t2) -> (new Tuple2<>(t1._1() + t2._1(), t1._2() + t2._2()));

        //  The logic is 'case (key, (num, count)) => (key, num / count)'
        Function<Tuple2<String, Tuple2<Double, Integer>>, ?> mapFunc = tuple -> new Tuple2<>(tuple._1(), tuple._2()._1() / tuple._2()._2());

        salaries
            .mapToPair(x -> new Tuple2<>(x[0], new Tuple2<>(Double.valueOf(x[1]), 1)))
            .reduceByKey(reduceFunc)
            .map(mapFunc)
            .collect()
            .forEach(System.out::println);

        JavaRDD<Row> rowRdd = salaries.map(x -> RowFactory.create(x[0], Integer.valueOf(x[1])));
        rowRdd.toDebugString();
        rowRdd.collect().forEach(System.out::println);

        StructField[] fields = new StructField[] {
            DataTypes.createStructField("name", StringType, true),
            DataTypes.createStructField("amount", IntegerType, true)};

        StructType salarySchema = DataTypes.createStructType(fields);

        Dataset<Row> df = new SQLContext(jsc).createDataFrame(rowRdd, salarySchema);

        df.groupBy("name")
            .agg(avg("amount"))
            .show();
    }
}
