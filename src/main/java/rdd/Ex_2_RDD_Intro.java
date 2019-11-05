package rdd;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import static rdd.SparkUtil.DATA_DIRECTORY;

public class Ex_2_RDD_Intro {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession();

        // Makes RDD based on Array
        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        List<Integer> r = Arrays.asList(1, 2, 3, 4, 5);

        // Creates RDD with 3 parts
        JavaRDD<Integer> ints = jsc.parallelize(r, 3);

        ints.saveAsTextFile(DATA_DIRECTORY + "/ints"); // works for windows well

        JavaRDD<Integer> cachedInts = jsc.textFile(DATA_DIRECTORY + "/ints")
            .map(Integer::valueOf)
            .cache();

        // Step 1: Transform each number to its square
        JavaRDD<Integer> squares = cachedInts.map(x -> x * x);

        System.out.println("--Squares--");
        squares.collect().forEach(System.out::println);

       /* // Step 2: Filter even numbers

        val even = squares.filter(x => x % 2 == 0)

        println("--Even numbers--")
        even.collect().foreach(println)

        // Step 3: print RDD metadata
        even.setName("Even numbers")
        println("Name is " + even.name + " id is " + even.id)
        println(even.toDebugString)

        println("Total multiplication is " + even.reduce((a, b) => a * b))

        // Step 4: Transform to PairRDD make keys 0 for even and 1 for odd numbers and
        val groups = cachedInts.map(x => if (x % 2 == 0) {
            (0, x)
        } else {
            (1, x)
        })

        println("--Groups--")
        println(groups.groupByKey.toDebugString)
        groups.groupByKey.collect.foreach(println)
        println(groups.countByKey)

        // Step 5: different actions
        println("--Different actions--")
        println("First elem is " + cachedInts.first)
        println("Total amount is " + cachedInts.count)
        println("Take 2")
        cachedInts.take(2).foreach(println)
        println("Take ordered 5")
        cachedInts.takeOrdered(5).foreach(println)*/

    }

    private static SparkSession getSparkSession() {
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
            .master("local[2]")
            .appName("RDD_Intro")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }
}
