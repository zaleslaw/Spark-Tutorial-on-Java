package ml.titanic;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Add two features: age and fare. Accuracy = 0.22
 */
public class Ex_8_Titanic_add_age_fare {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCasting(spark)
            .select("survived", "pclass", "sibsp", "parch", "sex", "embarked", "age", "fare");

        Dataset<Row>[] split = passengers.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> training = split[0].cache();
        Dataset<Row> test = split[1].cache();

        StringIndexer sexIndexer = new StringIndexer()
            .setInputCol("sex")
            .setOutputCol("sexIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        StringIndexer embarkedIndexer = new StringIndexer()
            .setInputCol("embarked")
            .setOutputCol("embarkedIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        Imputer imputer = new Imputer()
            .setInputCols(new String[]{"pclass", "sibsp", "parch", "age", "fare"})
            .setOutputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "age_imputed", "fare_imputed"})
            .setStrategy("mean");

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "age_imputed", "fare_imputed", "sexIndexed", "embarkedIndexed"})
            .setOutputCol("features");

        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{sexIndexer, embarkedIndexer, new TitanicUtils.DropSex(), imputer, assembler, trainer});

        PipelineModel model = pipeline.fit(training);

        Dataset<Row> rawPredictions = model.transform(test);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }
}
