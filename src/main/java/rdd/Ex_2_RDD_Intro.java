package rdd;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Simple transformations and actions on JavaRDD data.
 */
public class Ex_2_RDD_Intro {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("RDD_Intro");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        JavaRDD<Integer> cachedInts = jsc.textFile(DATA_DIRECTORY + "/ints")
            .map(Integer::valueOf)
            .cache();

        // Step 1: Transform each number to its square
        JavaRDD<Integer> squares = cachedInts.map(x -> x * x);

        println("--Squares--");
        squares.collect().forEach(System.out::println);

        // Step 2: Filter even numbers

        JavaRDD<Integer> even = squares.filter(x -> x % 2 == 0);

        println("--Even numbers--");
        even.collect().forEach(System.out::println);

        // Step 3: print RDD metadata
        even.setName("Even numbers");
        println("Name is " + even.name() + " id is " + even.id());
        println(even.toDebugString());

        println("Total multiplication is " + even.reduce((a, b) -> a * b));

        // Step 4: Transform to PairRDD make keys 0 for even and 1 for odd numbers and
        JavaPairRDD<Integer, Integer> groups = cachedInts.mapToPair(x -> {
            if (x % 2 == 0)
                return new Tuple2(0, x);
            else
                return new Tuple2(1, x);
        });

        println("--Groups--");
        println(groups.groupByKey().toDebugString());
        groups.groupByKey().collect().forEach(System.out::println);
        println(groups.countByKey().toString());

        // Step 5: different actions
        println("--Different actions--");
        println("First elem is " + cachedInts.first());
        println("Total amount is " + cachedInts.count());
        println("Take 2");
        cachedInts.take(2).forEach(System.out::println);
        println("Take ordered 5");
        cachedInts.takeOrdered(5).forEach(System.out::println);

    }
}
