#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fstream>

namespace py = pybind11;

namespace ote {

const double EPS = 1e-8;
const double EPS2 = 1e-5;
const double EPS3 = 1e-3;

inline int32_t sign(float x) {
  if (fabs(x) < EPS) {
    throw std::logic_error("computing sign of ~0");
  }
  if (x > 0) return 1;
  return -1;
}

class OTEstimators {
 public:
  using NumPyFloatArray = py::array_t<float, py::array::c_style>;
  using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

  using EigenVector = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Matrix = Eigen::Map<EigenMatrix>;

  OTEstimators() : stage(0) {}

  void load_vocabulary(NumPyFloatArray points) {
    if (stage != 0) {
      throw std::logic_error(
          "load_vocabulary() should be called once in the beginning");
    }
    stage = 1;
    py::buffer_info buf = points.request();
    if (buf.ndim != 2) {
      throw std::logic_error(
          "load_vocabulary() expects a two-dimensional NumPy array");
    }
    auto n = buf.shape[0];
    auto d = buf.shape[1];
    dictionary = std::make_unique<Matrix>(static_cast<float *>(buf.ptr), n, d); // vocab matrix
    auto cmin = std::numeric_limits<float>::max();
    auto cmax = std::numeric_limits<float>::min();
    for (ssize_t i = 0; i < n; ++i) {
      for (ssize_t j = 0; j < d; ++j) {
        cmin = std::min(cmin, (*dictionary)(i, j));
        cmax = std::max(cmax, (*dictionary)(i, j));
      }
    }
    auto delta = cmax - cmin; // the max value difference, d
    cmin -= delta; // the edge length of the cube is 2d
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<float> shift_gen(0.0, delta); // get random shift between (0,d)
    std::vector<std::pair<float, float>> bounding_box;
    for (ssize_t i = 0; i < d; ++i) {
      auto s = shift_gen(gen);
      bounding_box.push_back(std::make_pair(cmin + s, cmax + s)); // random shift
    }
    std::vector<int32_t> all;
    for (ssize_t i = 0; i < n; ++i) {
      all.push_back(i); // all vocab index
    }
    leaf.resize(n); // every word in vocab is a leaf
    build_quadtree(all, bounding_box, 0, -1);
    num_queries = 0;
    marked.resize(parents.size()); // all nodes + 1
    // all nodes in marked is -1
    for (auto &x : marked) {
      x = -1;
    }
    node_id.resize(parents.size());
    // py::print("Vocabulary loaded. Size:", buf.shape[0], ", Dimension:", buf.shape[1]);
  }

  void load_dataset(
      const std::vector<std::vector<std::pair<int32_t, float>>> &dataset) {
    if (stage != 1) {
      throw std::logic_error(
          "load_dataset() should be called once after calling "
          "load_vocabulary()");
    }
    stage = 2;
    if (dataset.empty()) {
      throw std::logic_error("the dataset can't be empty");
    }

    raw_dataset = dataset;
    // TODO: no idea what's this for
    // for (auto &measure : raw_dataset) {
    //   std::sort(measure.begin(), measure.end());
    // }

    dict_to_dataset0_index.resize(dataset[0].size());
    dict_to_dataset1_index.resize(dataset[1].size() + dataset[0].size());

    for (int i = 0; i < dataset[0].size(); ++i) {
      dict_to_dataset0_index[dataset[0][i].first] = i;
    }
    
    for (int i = 0; i < dataset[1].size(); ++i) {
      dict_to_dataset1_index[dataset[1][i].first] = i;
    }
    
    py::print("Dataset loaded. Size:", dataset.size());
  }

  std::pair<float, std::vector<std::vector<float>>> compute_flowtree_emd_between_dataset_points() {
    // py::print("Computing EMD between dataset points...");
    if (raw_dataset.size() != 2) {
      throw std::logic_error("The dataset must contain exactly two points for this operation.");
    }
    auto result = flowtree_query(raw_dataset[0], raw_dataset[1]);
    // py::print(raw_dataset[0]);
    // py::print(raw_dataset[1]);
    float emd = result.first;
    std::vector<std::vector<float>> flow_matrix = result.second;
    // py::print("Computed EMD done");
    return {emd, flow_matrix};
  }


 private:
  std::vector<int32_t> parents;
  std::vector<int32_t> leaf;
  std::vector<int32_t> marked;
  int32_t num_queries;
  std::vector<int32_t> node_id;
  std::vector<int32_t> id_node;
  std::vector<std::vector<int32_t>> subtree;
  std::vector<std::vector<std::pair<float, int32_t>>> excess;
  std::vector<float> delta_node;
  std::unique_ptr<Matrix> dictionary;
  std::vector<int32_t> unleaf;
  std::vector<std::vector<std::pair<int32_t, float>>> raw_dataset;
  int32_t stage;
  std::vector<int32_t> dict_to_dataset0_index;
  std::vector<int32_t> dict_to_dataset1_index;


  void build_quadtree(const std::vector<int32_t> &subset,
                      const std::vector<std::pair<float, float>> &bounding_box,
                      int32_t depth, int32_t parent) {
    // py::print("Inside build_quadtree function...");
    int32_t node_id(parents.size()); // already add how many nodes, use this as node id, 0~m
    parents.push_back(parent); // parents, the node id of all nodes, -1 is the parent of root

    if (subset.size() == 1) { // left one node
      leaf[subset[0]] = node_id; // leaf[leafid] = node_id, combine node_id to leaf
      return;
    }
    // actually, when leafid < samples_A.len, the leaf[leafid] is the sample belongs to A
    // leaf[leafid] is the node id of leafid sample

    int32_t d = dictionary->cols(); // dim
    std::vector<float> mid(d); // mid = d
    for (int32_t i = 0; i < d; ++i) {
      mid[i] = (bounding_box[i].first + bounding_box[i].second) / 2.0; // compute the origin
    }
    std::map<std::vector<uint8_t>, std::vector<int32_t>> parts;
    for (auto ind : subset) {
      std::vector<uint8_t> code((d + 7) / 8, 0);
      for (int32_t i = 0; i < d; ++i) {
        if ((*dictionary)(ind, i) > mid[i]) {
          code[i / 8] |= 1 << (i % 8);
        }
      }
      parts[code].push_back(ind);
    }
    std::vector<std::pair<float, float>> new_bounding_box(d);
    for (const auto &part : parts) {
      for (int32_t i = 0; i < d; ++i) {
        uint8_t bit = (part.first[i / 8] >> (i % 8)) & 1;
        if (bit) {
          new_bounding_box[i] = std::make_pair(mid[i], bounding_box[i].second);
        } else {
          new_bounding_box[i] = std::make_pair(bounding_box[i].first, mid[i]);
        }
      }
      build_quadtree(part.second, new_bounding_box, depth + 1, node_id);
    }
  }

  std::pair<float, std::vector<std::vector<float>>> flowtree_query(const std::vector<std::pair<int32_t, float>> &a,
                       const std::vector<std::pair<int32_t, float>> &b) {
    // py::print("Inside flowtree_query function...");
    int rows = a.size();
    int cols = b.size();
    std::vector<std::vector<float>> flow_matrix(rows, std::vector<float>(cols, 0.0f));
    int32_t num_nodes = 0;
    id_node.clear(); // id_node is a member vector

    // TODO later: find the subtree that contains the queries, in our situation, we don't actually need this step
    for (auto x : a) {
      auto id = leaf[x.first]; // the node id of a sample
      while (id != -1) { // already reached the root
        if (marked[id] != num_queries) { // not reached this node
          id_node.push_back(id); // put node id in id_node
          node_id[id] = num_nodes++; // TODO: node_id record the sequence of being reached?
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }
    for (auto x : b) {
      auto id = leaf[x.first];
      while (id != -1) {
        if (marked[id] != num_queries) {
          id_node.push_back(id);
          node_id[id] = num_nodes++;
        }
        marked[id] = num_queries;
        id = parents[id];
      }
    }
    if (static_cast<int32_t>(subtree.size()) < num_nodes) {
      subtree.resize(num_nodes);
    }
    for (int32_t i = 0; i < num_nodes; ++i) {
      subtree[i].clear();
    }
    
    for (int32_t i = 0; i < num_nodes; ++i) {
      int32_t u = parents[id_node[i]];
      if (u != -1) {
        subtree[node_id[u]].push_back(i); //record all the child
      }
    }
    if (static_cast<int32_t>(excess.size()) < num_nodes) {
      excess.resize(num_nodes);
    }
    delta_node.assign(num_nodes, 0.0);
    unleaf.resize(num_nodes);
    for (auto x : a) {
      delta_node[node_id[leaf[x.first]]] += x.second; // the leaf node's delta node record a.weight
      unleaf[node_id[leaf[x.first]]] = x.first; // record the sample id of unleaf
    }
    for (auto x : b) {
      delta_node[node_id[leaf[x.first]]] -= x.second;
      unleaf[node_id[leaf[x.first]]] = x.first;
    }
    // start from root node
    float res = run_query(0, node_id[0], flow_matrix);
    //  check unassigned node
    if (!excess[node_id[0]].empty()) {
      float unassigned = 0.0;
      for (auto x : excess[node_id[0]]) {
        unassigned += x.first;
      }
      if (unassigned > EPS2) {
        throw std::logic_error("too much unassigned flow");
      }
    }
    ++num_queries;
    return {res, flow_matrix};
  }

  float run_query(int32_t depth, int32_t nd, std::vector<std::vector<float>> &flow_matrix) {
    // py::print("Inside run_query function...");
    float res = 0.0;
    // go through the child of nd
    for (auto x : subtree[nd]) {
      res += run_query(depth + 1, x, flow_matrix);
    }
    excess[nd].clear();
    // if is leaf
    if (subtree[nd].empty()) {
      if (fabs(delta_node[nd]) > EPS) {
        excess[nd].push_back(std::make_pair(delta_node[nd], unleaf[nd]));
      }
    } else {
      for (auto x : subtree[nd]) {
        if (excess[x].empty()) {
          continue;
        }
        bool same = false;
        if (excess[nd].empty()) {
          same = true;
        } else if (sign(excess[x][0].first) == sign(excess[nd][0].first)) {
          same = true;
        }
        if (same) {
          for (auto y : excess[x]) {
            excess[nd].push_back(y);
          }
        } else {
          while (!excess[x].empty() && !excess[nd].empty()) {
            auto u = excess[nd].back();
            auto v = excess[x].back();
            float dist =
                (dictionary->row(u.second) - dictionary->row(v.second)).norm();
            int dataset0_index;
            int dataset1_index;
            if (sign(excess[x][0].first) > 0){
              dataset0_index = dict_to_dataset0_index[v.second];
              dataset1_index = dict_to_dataset1_index[u.second];
              // dataset0_index = v.second;
              // dataset1_index = u.second - flow_matrix.size();
            }
            else{
              dataset0_index = dict_to_dataset0_index[u.second];
              dataset1_index = dict_to_dataset1_index[v.second];
              // dataset0_index = u.second;
              // dataset1_index = v.second - flow_matrix.size();
            }
            // flow_matrix[u.second][v.second] = fabs(u.first);
            // py::print("Change!");
            if (fabs(u.first + v.first) < EPS) {
              excess[nd].pop_back();
              excess[x].pop_back();
              // py::print("Change1");
              // py::print("original res: ", res);
              res += dist * fabs(u.first);
              // py::print("after res: ", res);
              // py::print("dist: ", dist, "u.second: ", u.second, "v.second: ", v.second, "flow: ", fabs(u.first));
              // py::print("dataset0_index: ", dataset0_index, "dataset1_index: ", dataset1_index);
              flow_matrix[dataset0_index][dataset1_index] = fabs(u.first);
              // py::print(fabs(u.first));
            } else if (fabs(u.first) < fabs(v.first)) {
              excess[nd].pop_back();
              // py::print("Change2");
              // py::print("original res: ", res);
              excess[x].back().first += u.first;
              res += dist * fabs(u.first);
              // py::print("after res: ", res);
              // py::print("dist: ", dist, "u.second: ", u.second, "v.second: ", v.second, "flow: ", fabs(u.first));
              // py::print("dataset0_index: ", dataset0_index, "dataset1_index: ", dataset1_index);
              flow_matrix[dataset0_index][dataset1_index] = fabs(u.first);
            } else {
              excess[x].pop_back();
              excess[nd].back().first += v.first;
              // py::print("Change3");
              // py::print("original res: ", res);
              res += dist * fabs(v.first);
              // py::print("after res: ", res);
              // py::print("dist: ", dist, "u.second: ", u.second, "v.second: ", v.second, "flow: ", fabs(v.first));
              // py::print("dataset0_index: ", dataset0_index, "dataset1_index: ", dataset1_index);
              flow_matrix[dataset0_index][dataset1_index] = fabs(v.first);
            }
          }
          if (!excess[x].empty()) {
            excess[x].swap(excess[nd]);
          }
        }
      }
    }
    return res;
  }

  void check_measure(const std::vector<std::pair<int32_t, float>> &measure) {
    float sum = 0.0;
    auto n = dictionary->rows();
    for (auto &atom : measure) {
      if (atom.first < 0 || atom.first >= n) {
        throw std::logic_error("invalid index in the measure");
      }
      if (atom.second < -EPS) {
        throw std::logic_error("negative mass");
      }
      sum += atom.second;
    }
    if (fabs(sum - 1.0) > EPS3) {
      throw std::logic_error("the masses don't sum to 1");
    }
  }

  void check_stage() {
    if (stage != 2) {
      throw std::logic_error(
          "need to call load_vocabulary() and load_dataset() first");
    }
  }

  template <typename T>
  void check_dimension(T x) {
    auto buf = x.request();
    if (buf.ndim != 1) {
      throw std::logic_error(
          "input_ids, output_ids, output_scores must be one-dimensional");
    }
  }

  template <typename T>
  ssize_t get_length(T x) {
    return x.request().shape[0];
  }

  void check_input_output_arrays(NumPyIntArray input_ids,
                                 NumPyIntArray output_ids,
                                 NumPyFloatArray output_scores) {
    check_dimension(input_ids);
    check_dimension(output_ids);
    check_dimension(output_scores);
    auto l1 = get_length(input_ids);
    auto l2 = get_length(output_ids);
    auto l3 = get_length(output_scores);
    if (l2 != l3) {
      throw std::logic_error(
          "output_ids and output_scores must be of the same length");
    }
    if (l2 > l1) {
      throw std::logic_error(
          "output_ids and output_scores must be no longer than input_ids");
    }
    auto buf = static_cast<int32_t *>(input_ids.request().ptr);
    for (ssize_t i = 0; i < l1; ++i) {
      auto val = buf[i];
      if (val < 0 || val >= static_cast<int32_t>(raw_dataset.size())) {
        throw std::logic_error("input_ids contain an invalid index");
      }
    }
  }

  void check_input_output_arrays(NumPyIntArray input_ids,
                                 NumPyFloatArray input_scores,
                                 NumPyIntArray output_ids,
                                 NumPyFloatArray output_scores) {
    check_input_output_arrays(input_ids, output_ids, output_scores);
    check_dimension(input_scores);
    if (get_length(input_ids) != get_length(input_scores)) {
      throw std::logic_error(
          "input_ids and input_scores must be of the same length");
    }
  }
};
}  // namespace ote

PYBIND11_MODULE(ot_estimators, m) {
  using ote::OTEstimators;
  py::class_<OTEstimators>(m, "OTEstimators")
      .def(py::init<>())
      .def("load_vocabulary", &OTEstimators::load_vocabulary)
      .def("load_dataset", &OTEstimators::load_dataset)
      .def("compute_flowtree_emd_between_dataset_points", &OTEstimators::compute_flowtree_emd_between_dataset_points);
}
