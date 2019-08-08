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
package com.linkedin.photon.ml.function.glm

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.optimization.{OptimizerConfig, OptimizerType}
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}

/**
 * Unit tests for [[GLMLossFunction]].
 */
class GLMLossFunctionTest {

  import GLMLossFunctionTest._

  @DataProvider
  def coordinateOptimizationProblemProvider(): Array[Array[Any]] = {

    val optConfig = OptimizerConfig(OptimizerType.LBFGS, MAXIMUM_ITERATIONS, TOLERANCE)

    Array(
      Array(FixedEffectOptimizationConfiguration(optConfig)),
      Array(RandomEffectOptimizationConfiguration(optConfig)))
  }

  /**
   * Test that the [[ObjectiveFunction]] generated by the factory function returned by the [[GLMLossFunction]]
   * is of the appropriate type for the given coordinate optimization task.
   *
   * @param coordinateOptConfig The coordinate optimization task
   */
  @Test(dataProvider = "coordinateOptimizationProblemProvider")
  def testBuildFactory(coordinateOptConfig: CoordinateOptimizationConfiguration): Unit = {

    val objectiveFunction = GLMLossFunction.buildFactory(LOSS_FUNCTION, TREE_AGGREGATE_DEPTH)(coordinateOptConfig)()

    coordinateOptConfig match {
      case _: FixedEffectOptimizationConfiguration =>
        assertTrue(objectiveFunction.isInstanceOf[DistributedGLMLossFunction])

      case _: RandomEffectOptimizationConfiguration =>
        assertTrue(objectiveFunction.isInstanceOf[SingleNodeGLMLossFunction])

      case _ =>
        assertTrue(false)
    }
  }
}

object GLMLossFunctionTest {

  val LOSS_FUNCTION = LogisticLossFunction
  val MAXIMUM_ITERATIONS = 1
  val TOLERANCE = 2e-2
  val TREE_AGGREGATE_DEPTH = 3
}
