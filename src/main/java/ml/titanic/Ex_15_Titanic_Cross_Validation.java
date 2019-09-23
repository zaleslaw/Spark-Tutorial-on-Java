package ml.titanic;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Let's extract text features from "name" column. Divide each string on separate "names" and build hashingTF model
 * <p>
 * Remove stop words with StopWordsRemover
 * <p>
 * Select features with PCA. Accuracy = 0.213
 * <p>
 * Increasing amount of features with decreasing of accuracy is an example of overfit.
 */
public class Ex_15_Titanic_Cross_Validation {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCasting(spark)
            .select("survived", "pclass", "sibsp", "parch", "sex", "embarked", "age", "fare", "name");

        Dataset<Row>[] split = passengers.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> training = split[0].cache();
        Dataset<Row> test = split[1].cache();

        RegexTokenizer regexTokenizer = new RegexTokenizer()
            .setInputCol("name")
            .setOutputCol("name_parts")
            .setPattern("\\w+").setGaps(false);

        StopWordsRemover remover = new StopWordsRemover()
            .setStopWords(new String[]{"mr", "mrs", "miss", "master", "jr", "j", "c", "d"})
            .setInputCol("name_parts")
            .setOutputCol("filtered_name_parts");

        HashingTF hashingTF = new HashingTF()
            .setInputCol("filtered_name_parts")
            .setOutputCol("text_features")
            .setNumFeatures(1000);

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

        PolynomialExpansion polyExpansion = new PolynomialExpansion()
            .setInputCol("features")
            .setOutputCol("polyFeatures")
            .setDegree(2);

        // We should join together text features and number features into one vector
        VectorAssembler assembler2 = new VectorAssembler()
            .setInputCols(new String[]{"polyFeatures", "text_features"})
            .setOutputCol("joinedFeatures");

        MinMaxScaler scaler = new MinMaxScaler() // new MaxAbsScaler()
            .setInputCol("joinedFeatures")
            .setOutputCol("unnorm_features");

        Normalizer normalizer = new Normalizer()
            .setInputCol("unnorm_features")
            .setOutputCol("norm_features")
            .setP(1.0);

        PCA pca = new PCA()
            .setInputCol("norm_features")
            .setK(100)
            .setOutputCol("pca_features");

        RandomForestClassifier trainer = new RandomForestClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("pca_features")
                .setMaxDepth(20)
                .setNumTrees(200);

        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{
                regexTokenizer,
                remover,
                hashingTF,
                sexIndexer,
                embarkedIndexer,
                imputer,
                assembler,
                polyExpansion,
                assembler2,
                scaler,
                normalizer,
                pca,
                trainer});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("survived")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        ParamMap[] paramGrid = new ParamGridBuilder()
            .addGrid(hashingTF.numFeatures(), new int[]{100, 1000})
            .addGrid(pca.k(), new int[]{10, 100})
            .build();

        CrossValidator cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = cv.fit(training);

        System.out.println("---------- The best model's parameters are ----------");
        System.out.println("Num of features " + ((HashingTF)((PipelineModel)model.bestModel()).stages()[2]).getNumFeatures());
        System.out.println("Amount of components in PCA " + ((PCAModel)((PipelineModel)model.bestModel()).stages()[11]).getK());

        Dataset<Row> rawPredictions = model.transform(test);

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }
}
