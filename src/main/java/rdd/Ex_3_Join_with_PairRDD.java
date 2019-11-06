package rdd;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Join two PairRDDs.
 */
public class Ex_3_Join_with_PairRDD {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Join_with_PairRDD");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        // A few developers decided to commit something
        // Define pairs <Developer name, amount of committed core lines>
        List<Tuple2<String, Integer>> data = Arrays.asList(new Tuple2<>("Ivan", 240), new Tuple2<>("Petr", 39), new Tuple2<>("Elena", 290));
        JavaPairRDD<String, Integer> codeRows = jsc.parallelizePairs(data);

        // Let's calculate sum of code lines by developer
        codeRows.reduceByKey((x, y) -> x + y).collect().forEach(System.out::println);

        // Or group items to do something else
        codeRows.groupByKey().collect().forEach(System.out::println);

        // Don't forget about joins with preferred languages
        List<Tuple2<String, String>> profileData = Arrays.asList(new Tuple2<>("Ivan", "Java"), new Tuple2<>("Elena", "Scala"), new Tuple2<>("Petr", "Scala"));
        JavaPairRDD<String, String> programmerProfiles = jsc.parallelizePairs(profileData);

        JavaPairRDD<String, Tuple2<String, Integer>> joinResult = programmerProfiles.join(codeRows);
        println(joinResult.toDebugString());

        joinResult.collect().forEach(System.out::println);

        // also we can use special operator to group values from both rdd by key
        // also we sort in DESC order

        programmerProfiles.cogroup(codeRows).sortByKey(false).collect().forEach(System.out::println);

        // If required we can get amount of values by each key
        println(joinResult.countByKey().toString());

        // or get all values by specific key
        println(joinResult.lookup("Elena").toString());

        // codeRows keys only
        codeRows.keys().collect().forEach(System.out::println);

        // Print values only
        codeRows.values().collect().forEach(System.out::println);
    }
}
