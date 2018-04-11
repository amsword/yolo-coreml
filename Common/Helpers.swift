import Foundation
import UIKit
import CoreML
import Accelerate
// The labels for coco/voc office12
let labels = ["chair","phone","bottle","cup","clock","mouse","couch","keyboard","person","tvmonitor","laptop","pottedplant"]
// The labels for the 20 classes.
//let labels = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
//let labels = [ "accordion",  "airplane",  "ant",  "antelope",  "apple",  "armadillo",  "artichoke",  "axe",  "baby_bed",  "backpack",  "bagel",  "balance_beam",  "banana",  "band_aid",  "banjo",  "baseball",  "basketball",  "bathing_cap",  "beaker",  "bear",  "bee",  "bell_pepper",  "bench",  "bicycle",  "binder",  "bird",  "bookshelf",  "bow",  "bow_tie",  "bowl",  "brassiere",  "burrito",  "bus",  "butterfly",  "camel",  "can_opener",  "car",  "cart",  "cattle",  "cello",  "centipede",  "chain_saw",  "chair",  "chime",  "cocktail_shaker",  "coffee_maker",  "computer_keyboard",  "computer_mouse",  "corkscrew",  "cream",  "croquet_ball",  "crutch",  "cucumber",  "cup_or_mug",  "diaper",  "digital_clock",  "dishwasher",  "dog",  "domestic_cat",  "dragonfly",  "drum",  "dumbbell",  "electric_fan",  "elephant",  "face_powder",  "fig",  "filing_cabinet",  "flower_pot",  "flute",  "fox",  "french_horn",  "frog",  "frying_pan",  "giant_panda",  "goldfish",  "golf_ball",  "golfcart",  "guacamole",  "guitar",  "hair_dryer",  "hair_spray",  "hamburger",  "hammer",  "hamster",  "harmonica",  "harp",  "hat_with_a_wide_brim",  "head_cabbage",  "helmet",  "hippopotamus",  "horizontal_bar",  "horse",  "hotdog",  "iPod",  "isopod",  "jellyfish",  "koala_bear",  "ladle",  "ladybug",  "lamp",  "laptop",  "lemon",  "lion",  "lipstick",  "lizard",  "lobster",  "maillot",  "maraca",  "microphone",  "microwave",  "milk_can",  "miniskirt",  "monkey",  "motorcycle",  "mushroom",  "nail",  "neck_brace",  "oboe",  "orange",  "otter",  "pencil_box",  "pencil_sharpener",  "perfume",  "person",  "piano",  "pineapple",  "ping-pong_ball",  "pitcher",  "pizza",  "plastic_bag",  "plate_rack",  "pomegranate",  "popsicle",  "porcupine",  "power_drill",  "pretzel",  "printer",  "puck",  "punching_bag",  "purse",  "rabbit",  "racket",  "ray",  "red_panda",  "refrigerator",  "remote_control",  "rubber_eraser",  "rugby_ball",  "ruler",  "salt_or_pepper_shaker",  "saxophone",  "scorpion",  "screwdriver",  "seal",  "sheep",  "ski",  "skunk",  "snail",  "snake",  "snowmobile",  "snowplow",  "soap_dispenser",  "soccer_ball",  "sofa",  "spatula",  "squirrel",  "starfish",  "stethoscope",  "stove",  "strainer",  "strawberry",  "stretcher",  "sunglasses",  "swimming_trunks",  "swine",  "syringe",  "table",  "tape_player",  "tennis_ball",  "tick",  "tie",  "tiger",  "toaster",  "traffic_light",  "train",  "trombone",  "trumpet",  "turtle",  "tv_or_monitor",  "unicycle",  "vacuum",  "violin",  "volleyball",  "waffle_iron",  "washer",  "water_bottle",  "watercraft",  "whale",  "wine_bottle",  "zebra"]
let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }
}

/**
  Logistic sigmoid.
*/
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}
