package ml.titanic;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.DoubleType;

public class TitanicUtils {
    public static SparkSession getSparkSession(){
        //For windows only: don't forget to put winutils.exe to c:/bin folder
        System.setProperty("hadoop.home.dir", "c:\\");

        SparkSession spark = SparkSession.builder()
            .master("local")
            .appName("Spark_SQL")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }

    public static Dataset<Row> readPassengers(SparkSession spark){
        Dataset<Row> passengers = spark.read()
            .option("delimiter", ";")
            .option("inferSchema", "true")
            .option("header", "true")
            .csv("/home/zaleslaw/data/titanic.csv");

        passengers.printSchema();

        passengers.show();

        return passengers;
    }


    public static Dataset<Row> readPassengersWithCastingToDoubles(SparkSession spark) {
        Dataset<Row> passengers = spark.read()
            .option("delimiter", ";")
            .option("inferSchema", "true")
            .option("header", "true")
            .csv("/home/zaleslaw/data/titanic.csv");

        Dataset<Row> castedPassengers = passengers
            .withColumn("survived", new Column("survived").cast(DoubleType))
            .withColumn("pclass", new Column("pclass").cast(DoubleType))
            .withColumn("sibsp", new Column("sibsp").cast(DoubleType))
            .withColumn("parch", new Column("parch").cast(DoubleType));

        castedPassengers.printSchema();

        castedPassengers.show();

        return castedPassengers;
    }


    public static Dataset<Row> readPassengersWithCasting(SparkSession spark){
        Dataset<Row> passengers = spark.read()
            .option("delimiter", ";")
            .option("inferSchema", "true")
            .option("header", "true")
            .csv("/home/zaleslaw/data/titanic.csv");

        Dataset<Row> castedPassengers = passengers
            .withColumn("survived", new Column("survived").cast(DoubleType))
            .withColumn("pclass", new Column("pclass").cast(DoubleType))
            .withColumn("sibsp", new Column("sibsp").cast(DoubleType))
            .withColumn("parch", new Column("parch").cast(DoubleType))
            .withColumn("age", new Column("age").cast(DoubleType))
            .withColumn("fare", new Column("fare").cast(DoubleType));

        castedPassengers.printSchema();

        castedPassengers.show();

        return castedPassengers;
    }

    public static class DropSex extends Transformer {
        private long serialVersionUID = 5545470640951989469L;

        @Override public Dataset<Row> transform(Dataset<?> dataset) {
            Dataset<Row> result = dataset.drop("sex", "embarked"); // <============== drop columns to use Imputer
            System.out.println("DropSex is working");
            result.show();
            result.printSchema();
            return result;
        }

        @Override public StructType transformSchema(StructType schema) {
            return schema;
        }

        @Override public Transformer copy(ParamMap extra) {
            return null;
        }

        @Override public String uid() {
            return "CustomTransformer" + serialVersionUID;
        }
    }


    public static class Printer extends Transformer {

        private long serialVersionUID = 3345470640951989469L;

        @Override public Dataset<Row> transform(Dataset<?> dataset) {

            System.out.println(">>>>>>>>>> Printer output");
            dataset.show(false);
            dataset.printSchema();

            return dataset.toDF();
        }

        @Override public StructType transformSchema(StructType schema) {
            return schema;
        }

        @Override public Transformer copy(ParamMap extra) {
            return null;
        }

        @Override public String uid() {
            return "CustomTransformer" + serialVersionUID;
        }
    }
}
