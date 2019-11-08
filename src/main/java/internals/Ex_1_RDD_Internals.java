package internals;

import java.util.Arrays;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Shows how to print out information about partitions and change the shuffling streams of data via Custom Partitioner.
 */
public class Ex_1_RDD_Internals {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("RDD_Internals");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        JavaRDD<Integer> cachedInts = jsc.textFile(DATA_DIRECTORY + "/ints", 4)
            .map(Integer::valueOf)
            .cache();

        println("Amount of partitions " + cachedInts.getNumPartitions());

        // Step 1: Transform each number to its square
        JavaRDD<Integer> squares = cachedInts.map(x -> x * x);

        println("--Squares--");
        squares.collect().forEach(System.out::println);

        // Step 2: Filter even numbers
        JavaRDD<Integer> even = squares.filter(x -> x % 2 == 0);

        println("--Even numbers--");
        even.collect().forEach(System.out::println);

        even.coalesce(2).glom().collect().forEach(e -> System.out.println(Arrays.toString(e.toArray())));
        println("Amount of partitions " + even.partitions().size());

        even.coalesce(5).glom().collect().forEach(e -> System.out.println(Arrays.toString(e.toArray()))); // only 4 partitions due to docs
        println("Amount of partitions " + even.partitions().size());
        println(even.toDebugString());

        // Step - 3: Union with another RDD
        println("--Even and ints numbers");
        JavaRDD<Integer> union = even.union(cachedInts);
        union.repartition(7).glom().collect().forEach(e -> System.out.println(Arrays.toString(e.toArray())));
        println("Amount of partitions " + union.partitions().size());
        // yeah, real empty partitions

        // Step - 4: Custom partitioner
        println("Custom partitioner");
        union
            .mapToPair(e -> new Tuple2<>(e, e))
            .partitionBy(new EvenPartitioner(2))
            .glom()
            .collect()
            .forEach(e -> System.out.println(Arrays.toString(e.toArray())));
    }
}
