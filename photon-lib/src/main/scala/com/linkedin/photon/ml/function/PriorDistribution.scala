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
package com.linkedin.photon.ml.function

import breeze.linalg.{Vector, diag}
import breeze.numerics.abs
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper


trait PriorDistribution extends ObjectiveFunction {

  protected var _previousCoefficients: com.linkedin.photon.ml.model.Coefficients = _
  protected var _l1RegWeight: Double = 0D
  protected var _l2RegWeight: Double = 0D

  /**
    * Getter.
    *
    * @return The L21 regularization weight
    */
  def getL1RegularizationWeight: Double = _l1RegWeight

  /**
    * Getter.
    *
    * @return The L2 regularization weight
    */
  def getL2RegularizationWeight: Double = _l2RegWeight

  /**
    * Getter.
    *
    * @return The previous coefficients
    */
  def getPreviousCoefficients: com.linkedin.photon.ml.model.Coefficients = _previousCoefficients

  /**
    * Setter.
    *
    * @note This function definition uses the setter syntactic sugar trick. Statements like:
    *
    *    objectiveFunction.l1RegularizationWeight = 10
    *
    * will call this function.
    *
    * @param newL1RegWeight The new L1 regularization weight
    */
  protected[ml] def l1RegularizationWeight_= (newL1RegWeight: Double): Unit = _l1RegWeight = newL1RegWeight

  /**
    * Setter.
    *
    * @note This function definition uses the setter syntactic sugar trick. Statements like:
    *
    *    objectiveFunction.l2RegularizationWeight = 10
    *
    * will call this function.
    *
    * @param newL2RegWeight The new L2 regularization weight
    */
  protected[ml] def l2RegularizationWeight_= (newL2RegWeight: Double): Unit = _l2RegWeight = newL2RegWeight

  /**
    * Setter.
    *
    * @note This function definition uses the setter syntactic sugar trick. Statements like:
    *
    *    objectiveFunction.l2RegularizationWeight = 10
    *
    * will call this function.
    *
    * @param newPreviousCoefficients The new previous coefficients
    */
  protected[ml] def previousCoefficients_=
    (newPreviousCoefficients: com.linkedin.photon.ml.model.Coefficients): Unit
        = _previousCoefficients = newPreviousCoefficients

  /**
    * Compute the value of the function with L2 regularization over the given data for the given model coefficients.
    *
    * @param input The given data over which to compute the objective value
    * @param coefficients The model coefficients used to compute the function's value
    * @param normalizationContext The normalization context
    * @return The computed value of the function
    */
  abstract override protected[ml] def value(
    input: Data,
    coefficients: Coefficients,
    normalizationContext: BroadcastWrapper[NormalizationContext]): Double =
    super.value(input, coefficients, normalizationContext) + l2RegValue(convertToVector(coefficients))

  /**
    * Compute the L2 regularization value for the given model coefficients.
    *
    * @param coefficients The model coefficients
    * @return The L2 regularization value
    */
  protected def l1RegValue(coefficients: Vector[Double]): Double = {
    require(_previousCoefficients.hessianMatrixOption.isDefined, "Variance of previous batch should not be empty.")
    val normalizedCoefficients = (coefficients - _previousCoefficients.means) / diag(_previousCoefficients.hessianMatrixOption.get)
    _l2RegWeight * abs(normalizedCoefficients)
  }

  /**
    * Compute the L2 regularization value for the given model coefficients.
    *
    * @param coefficients The model coefficients
    * @return The L2 regularization value
    */
  protected def l2RegValue(coefficients: Vector[Double]): Double = {
    require(_previousCoefficients.hessianMatrixOption.isDefined, "Variance of previous batch should not be empty.")
    val normalizedCoefficients = (coefficients - _previousCoefficients.means) / diag(_previousCoefficients.hessianMatrixOption.get)
    _l2RegWeight * normalizedCoefficients.dot(normalizedCoefficients) / 2
  }
}

trait PriorDistributionDiff extends L2RegularizationDiff with PriorDistribution {

}

trait PriorDistributionTwiceDiff extends L2RegularizationTwiceDiff with PriorDistributionDiff {

}