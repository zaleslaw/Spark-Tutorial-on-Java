package sql;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.apache.spark.sql.types.DataTypes.StringType;
import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

public class Ex_1_CSV_to_Parquet_JSON {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("CSV_to_Parquet_JSON");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        // Step - 1: Extract the schema
        // Read CSV and automatically extract the schema

        Dataset<Row> stateNames = spark.read()
            .option("header", "true")
            .option("inferSchema", "true") // Id as int, count as int due to one extra pass over the data
            .csv(DATA_DIRECTORY + "/stateNames");

        stateNames.show();
        stateNames.printSchema();

        stateNames.write().parquet(DATA_DIRECTORY + "/stateNames");

        // Step - 2: In reality it can be too expensive and CPU-burst
        // If dataset is quite big, you can infer schema manually
        StructField[] fields = new StructField[] {
            DataTypes.createStructField("Id", IntegerType, true),
            DataTypes.createStructField("Name", StringType, true),
            DataTypes.createStructField("Year", IntegerType, true),
            DataTypes.createStructField("Gender", StringType, true),
            DataTypes.createStructField("Count", IntegerType, true)};

        StructType nationalNamesSchema = DataTypes.createStructType(fields);

        Dataset<Row> nationalNames = spark.read()
            .option("header", "true")
            .schema(nationalNamesSchema)
            .csv("/home/zaleslaw/data/NationalNames.csv");

        nationalNames.show();
        nationalNames.printSchema();
        nationalNames.write().json("/home/zaleslaw/data/nationalNames");
        // nationalNames.write.orc("/home/zaleslaw/data/nationalNames")
        // this is available only with HiveContext in opposite you will get an exception
        // Exception in thread "main" org.apache.spark.sql.AnalysisException: The ORC data source must be used with Hive support enabled;

        nationalNames.cache();

        // Step - 3: Simple dataframe operations

        // filter & select & orderBy
        nationalNames
            .where("Gender == 'M'")
            .select("Name", "Year", "Count")
            .orderBy("Name", "Year")
            .show(100);

        // Registered births by year in US since 1880
        nationalNames
            .groupBy("Year")
            .sum("Count").as("Sum")
            .orderBy("Year")
            .show(200);
    }
}
