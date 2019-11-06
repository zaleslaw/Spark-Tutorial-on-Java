package rdd;

import java.util.Arrays;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Set theory operations on the RDDs.
 */
public class Ex_5_Set_theory {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Set_theory");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        // Set Theory in Spark
        JavaRDD<String> jvmLanguages = jsc.parallelize(Arrays.asList("Scala", "Java", "Groovy", "Kotlin", "Ceylon"));
        JavaRDD<String> functionalLanguages = jsc.parallelize(Arrays.asList("Scala", "Kotlin", "JavaScript", "Haskell"));
        JavaRDD<String> webLanguages = jsc.parallelize(Arrays.asList("PHP", "Ruby", "Perl", "PHP", "JavaScript"));

        println("----Distinct----");
        JavaRDD<String> distinctLangs = webLanguages.union(jvmLanguages).distinct();
        println(distinctLangs.toDebugString());
        distinctLangs.collect().forEach(System.out::println);

        println("----Intersection----");
        JavaRDD<String> intersection = jvmLanguages.intersection(functionalLanguages);
        println(intersection.toDebugString());
        intersection.collect().forEach(System.out::println);

        println("----Substract----");
        JavaRDD<String> substraction = webLanguages.distinct().subtract(functionalLanguages);
        println(substraction.toDebugString());
        substraction.collect().forEach(System.out::println);

        println("----Cartesian----");
        JavaPairRDD<String, String> cartestian = webLanguages.distinct().cartesian(jvmLanguages);
        println(cartestian.toDebugString());
        cartestian.collect().forEach(System.out::println);
    }
}
