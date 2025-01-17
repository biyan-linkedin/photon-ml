/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.algorithm

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.{RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.PhotonNonBroadcast

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param dataset The training dataset
 * @param optimizationProblem The random effect optimization problem
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
    override protected val  dataset: RandomEffectDataset,
    protected val optimizationProblem: RandomEffectOptimizationProblem[Objective])
  extends Coordinate[RandomEffectDataset](dataset)
    with ModelProjection
    with RDDLike {

  //
  // Coordinate functions
  //

  /**
   * Update the coordinate with a new [[RandomEffectDataset]].
   *
   * @param dataset The updated [[RandomEffectDataset]]
   * @return A new coordinate with the updated [[RandomEffectDataset]]
   */
  override protected[algorithm] def updateCoordinateWithDataset(
      dataset: RandomEffectDataset): RandomEffectCoordinate[Objective] =
    new RandomEffectCoordinate(dataset, optimizationProblem)


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, OptimizationTracker) = {

    val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(dataset, optimizationProblem, None)

    (projectModelBackward(newModel), optimizationTracker)
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =

    model match {
      case randomEffectModel: RandomEffectModel =>
        val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(
          dataset,
          optimizationProblem,
          Some(projectModelForward(randomEffectModel)))

        (projectModelBackward(newModel), optimizationTracker)

      case _ =>
        throw new UnsupportedOperationException(
          s"Updating model of type ${model.getClass} in ${this.getClass} is not supported")
    }

  /**
   * Compute scores for the coordinate data using a given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case randomEffectModel: RandomEffectModel =>
      RandomEffectCoordinate.score(dataset, projectModelForward(randomEffectModel))

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblem.sparkContext

  /**
   * Assign a given name to the [[optimizationProblem]] [[RDD]].
   *
   * @param name The parent name for all [[RDD]] objects in this class
   * @return This object with the name of the [[optimizationProblem]] [[RDD]] assigned
   */
  override def setName(name: String): RandomEffectCoordinate[Objective] = {

    optimizationProblem.setName(name)

    this
  }

  /**
   * Set the persistence storage level of the [[optimizationProblem]] [[RDD]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of the [[optimizationProblem]] [[RDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectCoordinate[Objective] = {

    optimizationProblem.persistRDD(storageLevel)

    this
  }

  /**
   * Mark the [[optimizationProblem]] [[RDD]] as unused, and asynchronously remove all blocks for it from memory and
   * disk.
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] unpersisted
   */
  override def unpersistRDD(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.unpersistRDD()

    this
  }

  /**
   * Materialize the [[optimizationProblem]] [[RDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] materialized
   */
  override def materialize(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.materialize()

    this
  }
}

object RandomEffectCoordinate {

  /**
   * Helper function to construct [[RandomEffectCoordinate]] objects.
   *
   * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization
   *                               problems
   * @param randomEffectDataset The data on which to run the optimization algorithm
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType If and how coefficient variances should be computed
   * @return A new [[RandomEffectCoordinate]] object
   */
  protected[ml] def apply[RandomEffectObjective <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunction: RandomEffectObjective,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE): RandomEffectCoordinate[RandomEffectObjective] = {

    // Generate parameters of ProjectedRandomEffectCoordinate
    val randomEffectOptimizationProblem = buildRandomEffectOptimizationProblem(
      randomEffectDataset.projectors,
      configuration,
      objectiveFunction,
      glmConstructor,
      normalizationContext,
      varianceComputationType)

    new RandomEffectCoordinate(randomEffectDataset, randomEffectOptimizationProblem)
  }

  /**
   * Build a new [[RandomEffectOptimizationProblem]] for a [[RandomEffectCoordinate]] to optimize.
   *
   * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization
   *                               problems
   * @param linearSubspaceProjectorsRDD The per-entity [[LinearSubspaceProjector]] objects used to compress the
   *                                    per-entity feature spaces
   * @param configuration The optimization problem configuration
   * @param objectiveFunction The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType If and how coefficient variances should be computed
   * @return
   */
  private def buildRandomEffectOptimizationProblem[RandomEffectObjective <: SingleNodeObjectiveFunction](
      linearSubspaceProjectorsRDD: RDD[(REId, LinearSubspaceProjector)],
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunction: RandomEffectObjective,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE): RandomEffectOptimizationProblem[RandomEffectObjective] = {

    // Generate new NormalizationContext and SingleNodeOptimizationProblem objects
    val optimizationProblems = linearSubspaceProjectorsRDD
      .mapValues { projector =>
        val factors = normalizationContext.factorsOpt.map(factors => projector.projectForward(factors))
        val shiftsAndIntercept = normalizationContext
          .shiftsAndInterceptOpt
          .map { case (shifts, intercept) =>
            val newShifts = projector.projectForward(shifts)
            val newIntercept = projector.originalToProjectedSpaceMap(intercept)

            (newShifts, newIntercept)
          }
        val projectedNormalizationContext = new NormalizationContext(factors, shiftsAndIntercept)

        // TODO: Broadcast arguments to SingleNodeOptimizationProblem?
        SingleNodeOptimizationProblem(
          configuration,
          objectiveFunction,
          glmConstructor,
          PhotonNonBroadcast(projectedNormalizationContext),
          varianceComputationType)
      }

    new RandomEffectOptimizationProblem(optimizationProblems, glmConstructor)
  }

  /**
   * Train a new [[RandomEffectModel]] (i.e. run model optimization for each entity).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The training dataset
   * @param randomEffectOptimizationProblem The per-entity optimization problems
   * @param initialRandomEffectModelOpt An optional existing [[RandomEffectModel]] to use as a starting point for
   *                                    optimization
   * @return A (new [[RandomEffectModel]], optional optimization stats) tuple
   */
  protected[algorithm] def trainModel[Function <: SingleNodeObjectiveFunction](
      randomEffectDataset: RandomEffectDataset,
      randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function],
      initialRandomEffectModelOpt: Option[RandomEffectModel]): (RandomEffectModel, RandomEffectOptimizationTracker) = {

    // All 3 RDDs involved in these joins use the same partitioner
    val dataAndOptimizationProblems = randomEffectDataset
      .activeData
      .join(randomEffectOptimizationProblem.optimizationProblems)

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val (newModels, randomEffectOptimizationTracker) = initialRandomEffectModelOpt
      .map { randomEffectModel =>
        val modelsAndTrackers = randomEffectModel
          .modelsRDD
          .leftOuterJoin(dataAndOptimizationProblems)
          .mapValues {
            case (localModel, Some((localDataset, optimizationProblem))) =>
              val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
              val updatedModel = optimizationProblem.run(trainingLabeledPoints, localModel)
              val stateTrackers = optimizationProblem.getStatesTracker

              (updatedModel, Some(stateTrackers))

            case (localModel, _) =>
              (localModel, None)
          }
        modelsAndTrackers.persist(StorageLevel.MEMORY_ONLY_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.flatMap(_._2._2))

        (models, optimizationTracker)
      }
      .getOrElse {
        val modelsAndTrackers = dataAndOptimizationProblems.mapValues { case (localDataset, optimizationProblem) =>
          val trainingLabeledPoints = localDataset.dataPoints.map(_._2)
          val newModel = optimizationProblem.run(trainingLabeledPoints)
          val stateTrackers = optimizationProblem.getStatesTracker

          (newModel, stateTrackers)
        }
        modelsAndTrackers.persist(StorageLevel.MEMORY_ONLY_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.map(_._2._2))

        (models, optimizationTracker)
      }

    val newRandomEffectModel = new RandomEffectModel(
      newModels,
      randomEffectDataset.randomEffectType,
      randomEffectDataset.featureShardId)

    (newRandomEffectModel, randomEffectOptimizationTracker)
  }

  /**
   * Score a [[RandomEffectDataset]] using a given [[RandomEffectModel]].
   *
   * For information about the differences between active and passive data, see the [[RandomEffectDataset]]
   * documentation.
   *
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param randomEffectDataset The [[RandomEffectDataset]] to score
   * @param randomEffectModel The [[RandomEffectModel]] with which to score
   * @return The computed scores
   */
  protected[algorithm] def score(
      randomEffectDataset: RandomEffectDataset,
      randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    // Active data and models use the same partitioner, but scores need to use GameDatum partitioner
    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (localDataset, model)) =>
        localDataset.dataPoints.map { case (uniqueId, labeledPoint) =>
          (uniqueId, model.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)

    // Passive data already uses the GameDatum partitioner. Note that this code assumes few (if any) entities have a
    // passive dataset.
    val passiveDataREIds = randomEffectDataset.passiveDataREIds
    val modelsForPassiveData = randomEffectModel
      .modelsRDD
      .filter { case (reId, _) =>
        passiveDataREIds.value.contains(reId)
      }
      .collectAsMap()
    val passiveScores = randomEffectDataset
      .passiveData
      .mapValues { case (randomEffectId, labeledPoint) =>
        modelsForPassiveData(randomEffectId).computeScore(labeledPoint.features)
      }

    new CoordinateDataScores(activeScores ++ passiveScores)
  }
}
