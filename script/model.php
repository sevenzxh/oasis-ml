<?php
/**
 * Created by PhpStorm.
 * User: zhangxinghe
 * Date: 2019/2/12
 * Time: 15:17
 */

use Phpml\Classification\Classifier;
use Phpml\Classification\DecisionTree;
use Phpml\Classification\Ensemble\RandomForest;
use Phpml\Classification\SVC;
use Phpml\Classification\KNearestNeighbors;
use Phpml\Classification\NaiveBayes;
use Phpml\FeatureSelection\SelectKBest;
use Phpml\Metric\ClassificationReport;

require_once __DIR__ . '/../vendor/autoload.php';

error_reporting(E_ALL ^ E_WARNING ^ E_NOTICE);

// 训练数据
$train_samples = file_get_contents('../data/mpfen/train_samples.json');
$train_samples = \json_decode($train_samples, true);
$train_lables  = file_get_contents('../data/mpfen/train_lables.json');
$train_lables  = \json_decode($train_lables, true);

// 验证数据
$evaluate_samples = file_get_contents('../data/mpfen/evaluate_samples.json');
$evaluate_samples = \json_decode($evaluate_samples, true);
$evaluate_lables  = file_get_contents('../data/mpfen/evaluate_lables.json');
$evaluate_lables  = \json_decode($evaluate_lables, true);

// ANOVAF
echo "ANOVAF:\n";
$classifier = new SelectKBest(20);
$classifier->fit($train_samples, $train_lables);
var_dump($classifier->scores());

// KNN k=2
echo "KNN:\n";
$classifier = new KNearestNeighbors(3);
$rs         = trainPredictEvaluate($classifier, $train_samples, $train_lables, $evaluate_samples, $evaluate_lables);
echo "precision: {$rs['precision']}, recall: {$rs['recall']}\n\n\n";

// p(a|b) = p(b|a)p(a)/p(b)
echo "Bayes:\n";
$classifier = new NaiveBayes();
$rs         = trainPredictEvaluate($classifier, $train_samples, $train_lables, $evaluate_samples, $evaluate_lables);
echo "precision: {$rs['precision']}, recall: {$rs['recall']}\n\n\n";

// SVM
echo "SVM:\n";
$classifier = new SVC();
$rs         = trainPredictEvaluate($classifier, $train_samples, $train_lables, $evaluate_samples, $evaluate_lables);
echo "precision: {$rs['precision']}, recall: {$rs['recall']}\n\n\n";

// Random Forest
echo "Random Forest:\n";
$classifier = new RandomForest();
$rs         = trainPredictEvaluate($classifier, $train_samples, $train_lables, $evaluate_samples, $evaluate_lables);
echo "precision: {$rs['precision']}, recall: {$rs['recall']}\n\n\n";

function trainPredictEvaluate(Classifier $classifier,
                              $train_samples,
                              $train_lables,
                              $evaluate_samples,
                              $evaluate_lables)
{
    // 训练
    $classifier->train($train_samples, $train_lables);

    // 预测
    $predict_lables = $classifier->predict($evaluate_samples);

    // 验证:混淆矩阵
    $report = new ClassificationReport($evaluate_lables, $predict_lables);

    // return precison/recall
    return $report->getAverage();
}
