// mlx_mlir_parser.h - Native C++ MLIR text parser for standalone mode
// Replaces Python parser.py when MLX_STANDALONE is defined
// 
// NOTE: This header must be included AFTER the type definitions (MLXGraph, MLXOp)
// are already defined, since it uses those types but doesn't re-define them.

#ifndef MLX_MLIR_PARSER_H
#define MLX_MLIR_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <cstdlib>

// Forward declarations - types are defined in jax_mlx_pjrt.cpp
// This header expects MLXGraph and MLXOp to be already defined when it's included

namespace mlx_parser {

// Check if data is a portable artifact (bytecode) format
inline bool IsPortableArtifact(const char* data, size_t size) {
    // ML\xefR is the magic header for portable artifacts
    return size >= 4 && 
           data[0] == 'M' && data[1] == 'L' && 
           static_cast<unsigned char>(data[2]) == 0xef && data[3] == 'R';
}

// Lightweight MLIR text parser for StableHLO
class MLIRParser {
public:
    MLIRParser() : next_id_(0) {}
    
    bool parse(const std::string& text, MLXGraph& graph) {
        std::map<std::string, std::shared_ptr<MLXGraph>> dummy_functions;
        return parseAll(text, graph, dummy_functions);
    }
    
    // Parse all functions - main goes into graph, others go into functions map
    bool parseAll(const std::string& text, MLXGraph& graph, 
                  std::map<std::string, std::shared_ptr<MLXGraph>>& functions) {
        lines_.clear();
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) {
            lines_.push_back(line);
        }
        
        
        // Find all functions
        std::vector<std::pair<size_t, std::string>> func_locations;
        for (size_t i = 0; i < lines_.size(); ++i) {
            if (lines_[i].find("func.func") != std::string::npos) {
                // Extract function name
                std::string func_name;
                size_t at_pos = lines_[i].find('@');
                if (at_pos != std::string::npos) {
                    size_t name_end = lines_[i].find_first_of("( ", at_pos + 1);
                    if (name_end != std::string::npos) {
                        func_name = lines_[i].substr(at_pos + 1, name_end - at_pos - 1);
                    }
                }
                if (!func_name.empty()) {
                    func_locations.push_back({i, func_name});
                }
            }
        }
        
        if (func_locations.empty()) {
            std::cerr << "[MLX-PARSER] No functions found in MLIR" << std::endl;
            return false;
        }
        
        // Parse each function
        bool found_main = false;
        for (const auto& [line_num, func_name] : func_locations) {
            // Reset ssa_map for each function to avoid ID collisions
            ssa_map_.clear();
            int saved_next_id = next_id_;  // Save current ID counter
            
            if (func_name == "main" || 
                lines_[line_num].find("public @") != std::string::npos) {
                // This is the main function
                if (parseFunction(line_num, graph)) {
                    found_main = true;
                }
            } else {
                // This is a private function
                auto func_graph = std::make_shared<MLXGraph>();
                if (parseFunction(line_num, *func_graph)) {
                    functions[func_name] = func_graph;
                    if (debug_enabled()) {
                        std::cerr << "[MLX-PARSER] Parsed function: @" << func_name 
                                  << " (" << func_graph->nodes.size() << " nodes)" << std::endl;
                    }
                }
            }
        }
        
        if (!found_main) {
            std::cerr << "[MLX-PARSER] No main function found in MLIR" << std::endl;
            return false;
        }
        
        if (debug_enabled() && !functions.empty()) {
            std::cerr << "[MLX-PARSER] Parsed " << functions.size() << " additional functions" << std::endl;
        }
        
        return true;
    }
    
private:
    static bool debug_enabled() {
        static bool checked = false;
        static bool enabled = false;
        if (!checked) {
            enabled = std::getenv("MLX_PJRT_DEBUG") != nullptr;
            checked = true;
        }
        return enabled;
    }
    
    std::vector<std::string> lines_;
    int next_id_;
    std::unordered_map<std::string, int> ssa_map_;  // Maps %name to id
    
    bool parseFunction(size_t start_line, MLXGraph& graph) {
        // Parse function signature to get input arguments
        std::string& sig_line = lines_[start_line];
        
        // Extract arguments between ( and )
        size_t arg_start = sig_line.find('(');
        size_t arg_end = sig_line.find(')');
        if (arg_start != std::string::npos && arg_end != std::string::npos) {
            std::string args = sig_line.substr(arg_start + 1, arg_end - arg_start - 1);
            parseArguments(args, graph);
        }
        
        // Parse operations in the function body
        int brace_count = 0;
        bool in_body = false;
        std::string current_op_lines;
        
        // State for while-loop region parsing
        bool in_while_region = false;
        int while_region_depth = 0;       // brace depth within the while regions
        int while_region_index = -1;      // 0=cond, 1=body (do)
        std::string region_lines;         // accumulated lines for current region
        MLXOp* current_while_op = nullptr;
        
        // State for case region parsing (stablehlo.case / lax.cond / lax.switch)
        bool in_case_region = false;
        int case_region_depth = 0;      // brace depth within case regions
        MLXOp* current_case_op = nullptr;
        std::string case_region_lines;  // accumulated lines for current branch
        
        for (size_t i = start_line; i < lines_.size(); ++i) {
            const std::string& line = lines_[i];
            std::string trimmed = trim(line);
            
            // If we're inside a while region (cond/do block), handle specially
            if (in_while_region) {
                // If still waiting for "cond {" to start the first region
                if (while_region_index == -1) {
                    if (trimmed.find("cond") != std::string::npos &&
                        trimmed.find('{') != std::string::npos) {
                        while_region_depth = 1;
                        while_region_index = 0; // now in cond region
                    }
                    continue;
                }
                
                // Count braces in this line
                for (char c : trimmed) {
                    if (c == '{') while_region_depth++;
                    if (c == '}') while_region_depth--;
                }
                
                // Check for "} do {" transition (closes cond, opens body)
                // Only match at depth 1 (outer while boundary), not deeper nested whiles
                if (while_region_depth == 1 &&
                    (trimmed.find("} do {") != std::string::npos || 
                     trimmed.find("} do{") != std::string::npos)) {
                    // Parse the cond region we accumulated
                    if (current_while_op && while_region_index == 0) {
                        auto cond_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(region_lines, *cond_graph);
                        // Set input_ids from iterArg IDs
                        if (current_while_op->int_array_attrs.count("_iter_arg_ids")) {
                            auto& ids = current_while_op->int_array_attrs.at("_iter_arg_ids");
                            cond_graph->input_ids.assign(ids.begin(), ids.end());
                        }
                        current_while_op->subgraphs.push_back(cond_graph);
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] While cond region: " 
                                      << cond_graph->nodes.size() << " nodes, "
                                      << cond_graph->input_ids.size() << " inputs" << std::endl;
                        }
                    }
                    region_lines.clear();
                    while_region_index = 1;
                    continue;
                }
                
                // Check if we've exited all while regions (depth returned to 0)
                if (while_region_depth <= 0) {
                    // Parse the body (do) region
                    if (current_while_op && while_region_index == 1) {
                        auto body_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(region_lines, *body_graph);
                        // Set input_ids from iterArg IDs
                        if (current_while_op->int_array_attrs.count("_iter_arg_ids")) {
                            auto& ids = current_while_op->int_array_attrs.at("_iter_arg_ids");
                            body_graph->input_ids.assign(ids.begin(), ids.end());
                        }
                        current_while_op->subgraphs.push_back(body_graph);
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] While body region: " 
                                      << body_graph->nodes.size() << " nodes, "
                                      << body_graph->input_ids.size() << " inputs" << std::endl;
                        }
                    }
                    
                    region_lines.clear();
                    in_while_region = false;
                    current_while_op = nullptr;
                    while_region_index = -1;
                    continue;
                }
                
                // Handle stablehlo.return inside regions - accumulate for parseRegionOps
                if (trimmed.find("stablehlo.return") == 0 || trimmed.find("mhlo.return") == 0) {
                    region_lines += trimmed + "\n";
                    continue;
                }
                
                // Accumulate the line for region parsing
                region_lines += trimmed + "\n";
                continue;
            }
            
            // If we're inside a case region (branch blocks), handle specially
            if (in_case_region) {
                // Count braces in this line
                for (char c : trimmed) {
                    if (c == '{') case_region_depth++;
                    if (c == '}') case_region_depth--;
                }
                
                // Check for "}, {" transition (end of one branch, start of another)
                if ((trimmed.find("}, {") != std::string::npos || 
                     trimmed.find("},{") != std::string::npos) && 
                    case_region_depth == 1) {
                    // Parse the branch we accumulated
                    if (current_case_op) {
                        auto branch_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(case_region_lines, *branch_graph);
                        current_case_op->subgraphs.push_back(branch_graph);
                    }
                    case_region_lines.clear();
                    continue;
                }
                
                // Check if we've exited all case regions
                // The end pattern is "}) :" which terminates the case
                if (case_region_depth <= 0) {
                    // Parse the final branch 
                    if (current_case_op) {
                        auto branch_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(case_region_lines, *branch_graph);
                        current_case_op->subgraphs.push_back(branch_graph);
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] Case op: " 
                                      << current_case_op->subgraphs.size() << " branches" << std::endl;
                        }
                    }
                    
                    case_region_lines.clear();
                    in_case_region = false;
                    current_case_op = nullptr;
                    // Fix brace_count: the opening "({" incremented brace_count,
                    // but the closing "})" was consumed by the case handler without
                    // decrementing it. Re-synchronize the outer function's brace depth.
                    brace_count--;
                    continue;
                }
                
                // Accumulate the line for region parsing
                // (includes stablehlo.return and other ops)
                case_region_lines += trimmed + "\n";
                continue;
            }
            
            // Normal function body parsing
            // Track braces to know when we're in the function body
            for (char c : line) {
                if (c == '{') { brace_count++; in_body = true; }
                if (c == '}') brace_count--;
            }
            
            if (!in_body) continue;
            
            // Skip inline reducer(...) blocks within stablehlo.reduce ops
            // These have format: reducer(%arg1: ...) { ... stablehlo.return ... }
            // We still need to extract reduce_type from the body to set on the reduce op
            if (trimmed.find("reducer(") != std::string::npos || trimmed.find("reducer (") != std::string::npos) {
                // Collect body text to determine reduce_type
                std::string reducer_body = trimmed;
                int reducer_depth = 0;
                for (char c : trimmed) {
                    if (c == '{') reducer_depth++;
                    if (c == '}') reducer_depth--;
                }
                if (reducer_depth > 0) {
                    // Multi-line reducer block — collect text and skip
                    while (reducer_depth > 0 && ++i < lines_.size()) {
                        reducer_body += "\n" + lines_[i];
                        for (char c : lines_[i]) {
                            if (c == '{') reducer_depth++;
                            if (c == '}') { reducer_depth--; brace_count--; }
                        }
                    }
                }
                
                // Determine reduce_type from the collected body
                std::string reduce_type = "sum"; // default
                if (reducer_body.find("stablehlo.maximum") != std::string::npos || 
                    reducer_body.find("stablehlo.max") != std::string::npos) {
                    reduce_type = "max";
                } else if (reducer_body.find("stablehlo.minimum") != std::string::npos || 
                           reducer_body.find("stablehlo.min ") != std::string::npos) {
                    reduce_type = "min";
                } else if (reducer_body.find("stablehlo.multiply") != std::string::npos) {
                    reduce_type = "prod";
                } else if (reducer_body.find("stablehlo.add") != std::string::npos) {
                    reduce_type = "sum";
                } else if (reducer_body.find("stablehlo.or") != std::string::npos) {
                    reduce_type = "or";
                } else if (reducer_body.find("stablehlo.and") != std::string::npos) {
                    reduce_type = "and";
                } else if (reducer_body.find("stablehlo.select") != std::string::npos &&
                           reducer_body.find("stablehlo.compare") != std::string::npos) {
                    // Complex reducer with compare+select — likely argmax/argmin body
                    // Set to "sum" so the argmax pattern detector in ExecuteGraph can detect it
                    reduce_type = "sum";
                }
                
                // Set reduce_type on the most recent reduce op in the graph
                if (!graph.nodes.empty()) {
                    for (int j = graph.nodes.size() - 1; j >= 0; --j) {
                        if (graph.nodes[j].op_name == "stablehlo.reduce" || 
                            graph.nodes[j].op_name == "mhlo.reduce") {
                            graph.nodes[j].attributes["reduce_type"] = reduce_type;
                            if (debug_enabled()) {
                                std::cerr << "[MLX-PARSER] Set reduce_type=" << reduce_type 
                                          << " on inline reducer body" << std::endl;
                            }
                            break;
                        }
                    }
                }
                continue;
            }
            
            // Handle inline body regions for ops that use the ({ ^bb0(...): ... }) format.
            // This includes: select_and_scatter and reduce_window
            // We need to skip the body ops (they shouldn't become top-level graph nodes)
            // and extract relevant info (comparison_direction for select_and_scatter,
            // reduce_type for reduce_window).
            if (trimmed.find("^bb0") != std::string::npos || trimmed.find("^bb") != std::string::npos) {
                // The parent op may still be pending in current_op_lines.
                // Check if it's a select_and_scatter or reduce_window and flush it.
                bool is_sas = !current_op_lines.empty() && 
                    current_op_lines.find("select_and_scatter") != std::string::npos;
                bool is_rw = !current_op_lines.empty() &&
                    current_op_lines.find("reduce_window") != std::string::npos;
                
                if (is_sas || is_rw) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                
                // Find the parent op in graph.nodes
                int parent_idx = -1;
                std::string parent_type;
                for (int j = graph.nodes.size() - 1; j >= 0; --j) {
                    if (graph.nodes[j].op_name == "stablehlo.select_and_scatter" ||
                        graph.nodes[j].op_name == "mhlo.select_and_scatter") {
                        parent_idx = j;
                        parent_type = "sas";
                        break;
                    }
                    if (graph.nodes[j].op_name == "stablehlo.reduce_window" ||
                        graph.nodes[j].op_name == "mhlo.reduce_window") {
                        parent_idx = j;
                        parent_type = "rw";
                        break;
                    }
                    // Only look back a few ops
                    if ((int)graph.nodes.size() - 1 - j > 3) break;
                }
                
                if (parent_idx >= 0) {
                    // Collect all body text until the closing })
                    std::string body_text = trimmed;
                    
                    while (++i < lines_.size()) {
                        std::string body_line = trim(lines_[i]);
                        body_text += "\n" + body_line;
                        
                        for (char c : lines_[i]) {
                            if (c == '{') { brace_count++; in_body = true; }
                            if (c == '}') { brace_count--; }
                        }
                        
                        if (body_line.find("})") != std::string::npos) {
                            // End of all inline regions
                            if (parent_type == "sas") {
                                // Extract comparison direction for select_and_scatter
                                if (body_text.find("stablehlo.compare") != std::string::npos ||
                                    body_text.find("mhlo.compare") != std::string::npos) {
                                    if (body_text.find(" GE,") != std::string::npos ||
                                        body_text.find(" GE ") != std::string::npos) {
                                        graph.nodes[parent_idx].attributes["comparison_direction"] = "GE";
                                    } else if (body_text.find(" GT,") != std::string::npos ||
                                               body_text.find(" GT ") != std::string::npos) {
                                        graph.nodes[parent_idx].attributes["comparison_direction"] = "GT";
                                    } else if (body_text.find(" LE,") != std::string::npos ||
                                               body_text.find(" LE ") != std::string::npos) {
                                        graph.nodes[parent_idx].attributes["comparison_direction"] = "LE";
                                    } else if (body_text.find(" LT,") != std::string::npos ||
                                               body_text.find(" LT ") != std::string::npos) {
                                        graph.nodes[parent_idx].attributes["comparison_direction"] = "LT";
                                    }
                                }
                                if (debug_enabled()) {
                                    std::string dir = graph.nodes[parent_idx].attributes.count("comparison_direction")
                                        ? graph.nodes[parent_idx].attributes["comparison_direction"] : "UNKNOWN";
                                    std::cerr << "[MLX-PARSER] select_and_scatter: extracted comparison_direction="
                                              << dir << std::endl;
                                }
                            } else if (parent_type == "rw") {
                                // Extract reduce_type for reduce_window
                                if (body_text.find("stablehlo.maximum") != std::string::npos ||
                                    body_text.find("mhlo.maximum") != std::string::npos) {
                                    graph.nodes[parent_idx].attributes["reduce_type"] = "max";
                                } else if (body_text.find("stablehlo.minimum") != std::string::npos ||
                                           body_text.find("mhlo.minimum") != std::string::npos) {
                                    graph.nodes[parent_idx].attributes["reduce_type"] = "min";
                                } else if (body_text.find("stablehlo.add") != std::string::npos ||
                                           body_text.find("mhlo.add") != std::string::npos) {
                                    graph.nodes[parent_idx].attributes["reduce_type"] = "sum";
                                }
                                if (debug_enabled()) {
                                    std::string rt = graph.nodes[parent_idx].attributes.count("reduce_type")
                                        ? graph.nodes[parent_idx].attributes["reduce_type"] : "UNKNOWN";
                                    std::cerr << "[MLX-PARSER] reduce_window: extracted reduce_type="
                                              << rt << std::endl;
                                }
                            }
                            
                            // Extract output type from the }) line
                            // Format: }) : (input_types) -> output_type
                            // The parent op was parsed early (before the body), so it
                            // may not have output type info. Extract it from this closing line.
                            size_t arrow_pos = body_line.find("->");
                            if (arrow_pos != std::string::npos && 
                                graph.nodes[parent_idx].output_shapes.empty()) {
                                std::string output_type_str = trim(body_line.substr(arrow_pos + 2));
                                std::regex tensor_regex(R"(tensor<([^>]+)>)");
                                std::sregex_iterator type_iter(output_type_str.begin(), output_type_str.end(), tensor_regex);
                                std::sregex_iterator type_end;
                                while (type_iter != type_end) {
                                    std::string shape_dtype = (*type_iter)[1].str();
                                    graph.nodes[parent_idx].output_shapes.push_back(parseShape(shape_dtype));
                                    graph.nodes[parent_idx].output_dtypes.push_back(parseDtype(shape_dtype));
                                    ++type_iter;
                                }
                                if (debug_enabled()) {
                                    std::cerr << "[MLX-PARSER] " << graph.nodes[parent_idx].op_name 
                                              << ": extracted output shape from }) line" << std::endl;
                                }
                            }
                            break;
                        }
                    }
                    continue;
                }
            }
            
            // Handle end of function
            if (brace_count == 0) {
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                break;
            }
            
            // Skip empty lines and comments
            if (trimmed.empty() || trimmed[0] == '/') continue;
            
            // Handle func.return - explicit terminator
            if (trimmed.find("func.return") == 0 || trimmed.find("return") == 0) {
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                parseReturn(trimmed, graph);
                continue;
            }
            
            // Check if this line starts a stablehlo.while op
            // Format: %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %cst) : ...
            if (trimmed.find("stablehlo.while") != std::string::npos ||
                trimmed.find("mhlo.while") != std::string::npos) {
                // Finish any pending op before while
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                
                // Parse the while op line
                parseWhileOp(trimmed, graph);
                current_while_op = &graph.nodes.back();
                
                // Check if this line already contains "cond {"
                if (trimmed.find("cond") != std::string::npos && 
                    trimmed.find('{') != std::string::npos) {
                    // The "cond {" is on the SAME line as the while op
                    in_while_region = true;
                    while_region_depth = 1; // we're inside {
                    while_region_index = 0; // cond region
                    region_lines.clear();
                } else {
                    // The "cond {" might be on the NEXT line
                    in_while_region = true;
                    while_region_depth = 0;
                    while_region_index = -1; // waiting for "cond {"
                    region_lines.clear();
                }
                continue;
            }

            // Check if this line starts a stablehlo.case op
            // Format: %0 = "stablehlo.case"(%c) ({
            //    or:  %0 = stablehlo.case %c : (tensor<i32>) -> tensor<f32> ({
            if (trimmed.find("stablehlo.case") != std::string::npos ||
                trimmed.find("mhlo.case") != std::string::npos) {
                // Finish any pending op before case
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                
                // Parse the case op line
                parseCaseOp(trimmed, graph);
                current_case_op = &graph.nodes.back();
                
                // The line typically contains "({" — start tracking at depth 1
                int initial_depth = 0;
                for (char c : trimmed) {
                    if (c == '{') initial_depth++;
                    if (c == '}') initial_depth--;
                }
                
                in_case_region = true;
                case_region_depth = initial_depth;
                case_region_lines.clear();
                continue;
            }

            // Check if this line starts a NEW operation assignment: %name = ...
            // Must have % and =
            size_t eq_pos = trimmed.find('=');
            bool starts_new_op = false;
            
            if (eq_pos != std::string::npos) {
                std::string lhs = trimmed.substr(0, eq_pos);
                if (lhs.find('%') != std::string::npos) {
                    // Check if it's an argument decl (skip) or op assignment
                    // Argument decl: %arg0: tensor... (has colon, tensor type)
                    // Op assignment: %0 = ... (no type info before =)
                    // But wait, multi-output op: %0:2 = ...
                    // Check if LHS has "tensor" keyword -> argument
                    if (lhs.find("tensor") == std::string::npos) {
                        starts_new_op = true;
                    }
                }
            }
            
            if (starts_new_op) {
                // If we were building an op, parse it now
                if (!current_op_lines.empty()) {
                    parseOperation(current_op_lines, graph);
                    current_op_lines.clear();
                }
                // Start new op
                current_op_lines = trimmed;
            } else {
                // Continuation line (attributes etc.)
                // Only append if we are currently building an op
                if (!current_op_lines.empty()) {
                    current_op_lines += " " + trimmed;
                }
            }
        }
        
        return true;
    }
    
    // Parse a stablehlo.case op line
    // Format 1: %0 = "stablehlo.case"(%c) ({
    // Format 2: %0 = stablehlo.case %c : (tensor<i32>) -> tensor<f32> ({
    void parseCaseOp(const std::string& line, MLXGraph& graph) {
        MLXOp op;
        op.op_name = line.find("mhlo.case") != std::string::npos ? "mhlo.case" : "stablehlo.case";
        
        // Parse LHS: %0 = ...
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string lhs = trim(line.substr(0, eq_pos));
            
            // Check for multi-output (%0:N)
            size_t colon_pos = lhs.find(':');
            if (colon_pos != std::string::npos) {
                std::string base = lhs.substr(0, colon_pos);
                try {
                    int count = std::stoi(lhs.substr(colon_pos + 1));
                    for (int i = 0; i < count; ++i) {
                        int id = next_id_++;
                        ssa_map_[base + "#" + std::to_string(i)] = id;
                        op.outputs.push_back(id);
                    }
                } catch (...) {
                    int id = next_id_++;
                    ssa_map_[lhs] = id;
                    op.outputs.push_back(id);
                }
            } else {
                int id = next_id_++;
                ssa_map_[lhs] = id;
                op.outputs.push_back(id);
            }
        }
        
        // Parse input: the index argument
        // Format 1: "stablehlo.case"(%c) — index in parens after quoted name
        // Format 2: stablehlo.case %c : — index after op name
        std::regex input_regex(R"(%([\w]+))");
        std::string rhs = (eq_pos != std::string::npos) ? line.substr(eq_pos + 1) : line;
        
        // Find the first %name after "case"
        size_t case_pos = rhs.find("case");
        if (case_pos != std::string::npos) {
            std::string after_case = rhs.substr(case_pos);
            auto it = std::sregex_iterator(after_case.begin(), after_case.end(), input_regex);
            if (it != std::sregex_iterator()) {
                std::string name = "%" + (*it)[1].str();
                if (ssa_map_.count(name)) {
                    op.inputs.push_back(ssa_map_[name]);
                }
            }
        }
        
        graph.nodes.push_back(op);
    }
    
    void parseArguments(const std::string& args, MLXGraph& graph) {
        // Parse: %arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>
        std::regex arg_regex(R"(%(\w+)\s*:\s*tensor<([^>]+)>)");
        std::sregex_iterator iter(args.begin(), args.end(), arg_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::smatch match = *iter;
            std::string name = match[1].str();
            std::string shape_dtype = match[2].str();
            
            int id = next_id_++;
            ssa_map_["%" + name] = id;
            graph.input_ids.push_back(id);
            
            // Parse shape
            std::vector<int> shape = parseShape(shape_dtype);
            graph.input_shapes.push_back(shape);
            
            ++iter;
        }
    }
    
    // Parse a stablehlo.while op line
    // Format: %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %cst) : tensor<i32>, tensor<f32>
    void parseWhileOp(const std::string& line, MLXGraph& graph) {
        MLXOp op;
        op.op_name = line.find("mhlo.while") != std::string::npos ? "mhlo.while" : "stablehlo.while";
        
        // Parse LHS: %0:2 = ...
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string lhs = line.substr(0, eq_pos);
            // Extract result name and count
            std::regex result_regex(R"(%([\w]+)(?::(\d+))?)");
            std::smatch match;
            if (std::regex_search(lhs, match, result_regex)) {
                std::string name = match[1].str();
                int count = match[2].matched ? std::stoi(match[2].str()) : 1;
                for (int r = 0; r < count; ++r) {
                    int id = next_id_++;
                    op.outputs.push_back(id);
                    if (count > 1) {
                        ssa_map_["%" + name + "#" + std::to_string(r)] = id;
                    } else {
                        ssa_map_["%" + name] = id;
                    }
                }
            }
        }
        
        // Parse iterArg mappings: (%iterArg = %c, %iterArg_0 = %cst)
        // These define the while's inputs AND the region arguments
        size_t while_paren_start = line.find("while(");
        if (while_paren_start == std::string::npos) while_paren_start = line.find("while (");
        if (while_paren_start != std::string::npos) {
            size_t paren_start = line.find('(', while_paren_start);
            // Find matching closing paren
            size_t paren_end = std::string::npos;
            int depth = 0;
            for (size_t p = paren_start; p < line.size(); ++p) {
                if (line[p] == '(') depth++;
                if (line[p] == ')') { depth--; if (depth == 0) { paren_end = p; break; } }
            }
            
            if (paren_start != std::string::npos && paren_end != std::string::npos) {
                std::string args_str = line.substr(paren_start + 1, paren_end - paren_start - 1);
                
                // Parse each: %iterArg = %c, %iterArg_0 = %cst
                std::regex iter_arg_regex(R"(%([\w]+)\s*=\s*%([\w]+))");
                auto iter_begin = std::sregex_iterator(args_str.begin(), args_str.end(), iter_arg_regex);
                auto iter_end = std::sregex_iterator();
                
                for (auto it = iter_begin; it != iter_end; ++it) {
                    std::string iter_name = (*it)[1].str();
                    std::string operand_name = (*it)[2].str();
                    
                    // Look up operand in ssa_map to get input ID
                    std::string operand_key = "%" + operand_name;
                    int input_id = -1;
                    if (ssa_map_.count(operand_key)) {
                        input_id = ssa_map_[operand_key];
                    }
                    
                    if (input_id >= 0) {
                        op.inputs.push_back(input_id);
                    }
                    
                    // Create IDs for the iterArg names - these will be used as 
                    // the region input IDs
                    int iter_id = next_id_++;
                    ssa_map_["%" + iter_name] = iter_id;
                    // Store the iter arg IDs for the regions to use
                    op.int_array_attrs["_iter_arg_ids"].push_back(iter_id);
                }
            }
        }
        
        if (debug_enabled()) {
            std::cerr << "[MLX-PARSER] While op: " << op.outputs.size() << " outputs, " 
                      << op.inputs.size() << " inputs";
            std::cerr << " out_ids=[";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << op.outputs[i];
            }
            std::cerr << "] in_ids=[";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << op.inputs[i];
            }
            std::cerr << "]" << std::endl;
        }
        
        graph.nodes.push_back(std::move(op));
    }
    
    // Parse operations inside a while-loop region (cond or do block)
    // region_text contains the accumulated lines from the region
    // Supports nested stablehlo.while and stablehlo.case ops
    void parseRegionOps(const std::string& region_text, MLXGraph& region_graph) {
        // The region uses the same ssa_map as the parent - iterArg names are already mapped
        if (debug_enabled()) {
            std::cerr << "[MLX-PARSER] parseRegionOps text: '''" << region_text << "'''" << std::endl;
        }
        
        // Split into lines for processing
        std::vector<std::string> region_lines;
        std::istringstream stream(region_text);
        std::string line;
        while (std::getline(stream, line)) {
            region_lines.push_back(trim(line));
        }
        
        // State for nested while-loop region parsing
        bool in_nested_while = false;
        int nested_while_depth = 0;
        int nested_while_region_index = -1;  // 0=cond, 1=body
        std::string nested_region_lines;
        MLXOp* nested_while_op = nullptr;
        
        for (size_t i = 0; i < region_lines.size(); ++i) {
            const std::string& trimmed = region_lines[i];
            if (trimmed.empty()) continue;
            
            // Handle nested while region accumulation
            if (in_nested_while) {
                if (nested_while_region_index == -1) {
                    if (trimmed.find("cond") != std::string::npos &&
                        trimmed.find('{') != std::string::npos) {
                        nested_while_depth = 1;
                        nested_while_region_index = 0;
                    }
                    continue;
                }
                
                for (char c : trimmed) {
                    if (c == '{') nested_while_depth++;
                    if (c == '}') nested_while_depth--;
                }
                
                // "} do {" transition
                if (trimmed.find("} do {") != std::string::npos ||
                    trimmed.find("} do{") != std::string::npos) {
                    if (nested_while_op && nested_while_region_index == 0) {
                        auto cond_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(nested_region_lines, *cond_graph);
                        if (nested_while_op->int_array_attrs.count("_iter_arg_ids")) {
                            auto& ids = nested_while_op->int_array_attrs.at("_iter_arg_ids");
                            cond_graph->input_ids.assign(ids.begin(), ids.end());
                        }
                        nested_while_op->subgraphs.push_back(cond_graph);
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] Nested while cond region: "
                                      << cond_graph->nodes.size() << " nodes, "
                                      << cond_graph->input_ids.size() << " inputs" << std::endl;
                        }
                    }
                    nested_region_lines.clear();
                    nested_while_region_index = 1;
                    continue;
                }
                
                // Exited all while regions
                if (nested_while_depth <= 0) {
                    if (nested_while_op && nested_while_region_index == 1) {
                        auto body_graph = std::make_shared<MLXGraph>();
                        parseRegionOps(nested_region_lines, *body_graph);
                        if (nested_while_op->int_array_attrs.count("_iter_arg_ids")) {
                            auto& ids = nested_while_op->int_array_attrs.at("_iter_arg_ids");
                            body_graph->input_ids.assign(ids.begin(), ids.end());
                        }
                        nested_while_op->subgraphs.push_back(body_graph);
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] Nested while body region: "
                                      << body_graph->nodes.size() << " nodes, "
                                      << body_graph->input_ids.size() << " inputs" << std::endl;
                        }
                    }
                    nested_region_lines.clear();
                    in_nested_while = false;
                    nested_while_op = nullptr;
                    nested_while_region_index = -1;
                    continue;
                }
                
                // Accumulate line for nested region
                nested_region_lines += trimmed + "\n";
                continue;
            }
            
            // Handle stablehlo.return - defines region outputs
            if (trimmed.find("stablehlo.return") == 0 || trimmed.find("mhlo.return") == 0) {
                size_t colon_pos = trimmed.find(" : ");
                std::string args_part = (colon_pos != std::string::npos) 
                    ? trimmed.substr(0, colon_pos)
                    : trimmed;
                
                size_t space_pos = args_part.find(' ');
                if (space_pos != std::string::npos) {
                    std::string vals = args_part.substr(space_pos + 1);
                    std::regex val_regex(R"(%([\w#]+))");
                    auto it = std::sregex_iterator(vals.begin(), vals.end(), val_regex);
                    auto end = std::sregex_iterator();
                    for (; it != end; ++it) {
                        std::string name = "%" + (*it)[1].str();
                        if (ssa_map_.count(name)) {
                            region_graph.output_ids.push_back(ssa_map_[name]);
                        }
                    }
                }
                continue;
            }
            
            // Check for nested stablehlo.while
            if (trimmed.find("stablehlo.while") != std::string::npos ||
                trimmed.find("mhlo.while") != std::string::npos) {
                parseWhileOp(trimmed, region_graph);
                nested_while_op = &region_graph.nodes.back();
                
                if (trimmed.find("cond") != std::string::npos &&
                    trimmed.find('{') != std::string::npos) {
                    in_nested_while = true;
                    nested_while_depth = 1;
                    nested_while_region_index = 0;
                    nested_region_lines.clear();
                } else {
                    in_nested_while = true;
                    nested_while_depth = 0;
                    nested_while_region_index = -1;
                    nested_region_lines.clear();
                }
                continue;
            }
            
            // Check for nested stablehlo.case (lax.cond/lax.switch)
            if (trimmed.find("stablehlo.case") != std::string::npos ||
                trimmed.find("mhlo.case") != std::string::npos) {
                parseCaseOp(trimmed, region_graph);
                MLXOp* nested_case_op = &region_graph.nodes.back();
                
                // Count initial brace depth from this line
                int case_depth = 0;
                for (char c : trimmed) {
                    if (c == '{') case_depth++;
                    if (c == '}') case_depth--;
                }
                
                if (case_depth > 0) {
                    // Accumulate case branch lines
                    std::string case_branch_lines;
                    
                    for (++i; i < region_lines.size() && case_depth > 0; ++i) {
                        const std::string& case_line = region_lines[i];
                        
                        for (char c : case_line) {
                            if (c == '{') case_depth++;
                            if (c == '}') case_depth--;
                        }
                        
                        // Check for "}, {" transition between branches
                        if ((case_line.find("}, {") != std::string::npos ||
                             case_line.find("},{") != std::string::npos) &&
                            case_depth == 1) {
                            auto branch_graph = std::make_shared<MLXGraph>();
                            parseRegionOps(case_branch_lines, *branch_graph);
                            nested_case_op->subgraphs.push_back(branch_graph);
                            case_branch_lines.clear();
                            continue;
                        }
                        
                        // Exited all case regions
                        if (case_depth <= 0) {
                            auto branch_graph = std::make_shared<MLXGraph>();
                            parseRegionOps(case_branch_lines, *branch_graph);
                            nested_case_op->subgraphs.push_back(branch_graph);
                            if (debug_enabled()) {
                                std::cerr << "[MLX-PARSER] Nested case op: "
                                          << nested_case_op->subgraphs.size() << " branches" << std::endl;
                            }
                            break;
                        }
                        
                        case_branch_lines += case_line + "\n";
                    }
                }
                continue;
            }
            
            // Check for ops with inline brace blocks (reduce_window, sort, reduce)
            // These have format: %x = "stablehlo.reduce_window"(...) <{attrs}> ({
            //   ... reducer body ...
            // }) : ...
            // We need to skip the brace block and extract reduce_type from it
            {
                int line_brace_depth = 0;
                for (char c : trimmed) {
                    if (c == '{') line_brace_depth++;
                    if (c == '}') line_brace_depth--;
                }
                
                if (line_brace_depth > 0) {
                    // This line opens braces that aren't closed on the same line
                    // Accumulate until braces balance to get the full op text
                    std::string accumulated_body;
                    int depth = line_brace_depth;
                    
                    for (++i; i < region_lines.size() && depth > 0; ++i) {
                        const std::string& brace_line = region_lines[i];
                        for (char c : brace_line) {
                            if (c == '{') depth++;
                            if (c == '}') depth--;
                        }
                        accumulated_body += brace_line + "\n";
                    }
                    // Back up one since the for loop will increment
                    if (i > 0) --i;
                    
                    // Parse the op from the first line (which has the op name, inputs, attrs)
                    parseOperation(trimmed, region_graph);
                    
                    // For reduce_window: extract reduce_type from the accumulated body
                    MLXOp& last_op = region_graph.nodes.back();
                    if ((last_op.op_name == "stablehlo.reduce_window" || last_op.op_name == "mhlo.reduce_window") &&
                        !last_op.attributes.count("reduce_type")) {
                        if (accumulated_body.find("maximum") != std::string::npos ||
                            accumulated_body.find("maxf") != std::string::npos) {
                            last_op.attributes["reduce_type"] = "max";
                        } else if (accumulated_body.find("minimum") != std::string::npos ||
                                   accumulated_body.find("minf") != std::string::npos) {
                            last_op.attributes["reduce_type"] = "min";
                        } else {
                            last_op.attributes["reduce_type"] = "sum";
                        }
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] Nested reduce_window: extracted reduce_type="
                                      << last_op.attributes["reduce_type"] << std::endl;
                        }
                    }
                    
                    // For sort: extract comparison direction from the accumulated body
                    if ((last_op.op_name == "stablehlo.sort" || last_op.op_name == "mhlo.sort") &&
                        !last_op.attributes.count("sort_direction")) {
                        // stablehlo.compare LT = ascending, GT = descending
                        if (accumulated_body.find(" LT") != std::string::npos) {
                            last_op.attributes["sort_direction"] = "ascending";
                        } else if (accumulated_body.find(" GT") != std::string::npos) {
                            last_op.attributes["sort_direction"] = "descending";
                        } else {
                            last_op.attributes["sort_direction"] = "ascending";
                        }
                        if (debug_enabled()) {
                            std::cerr << "[MLX-PARSER] Nested sort: direction="
                                      << last_op.attributes["sort_direction"] << std::endl;
                        }
                    }
                    
                    continue;
                }
            }
            
            // Parse as a normal operation (single-line ops)
            parseOperation(trimmed, region_graph);
        }
    }
    
    void parseOperation(const std::string& line, MLXGraph& graph) {
        MLXOp op;
        
        // Parse: %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
        // Or: %result:2 = stablehlo.split %input ... (multiple results)
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) return;
        
        // Parse outputs (left of =)
        std::string outputs_str = trim(line.substr(0, eq_pos));
        std::vector<std::string> output_names;
        
        // Check for multiple outputs (%result:2)
        size_t output_colon_pos = outputs_str.find(':');
        if (output_colon_pos != std::string::npos && 
            outputs_str.find("tensor") == std::string::npos) {
            // Multi-output case: %17:2 creates %17#0, %17#1
            std::string base_name = outputs_str.substr(0, output_colon_pos);
            std::string count_str = outputs_str.substr(output_colon_pos + 1);
            try {
                int num_outputs = std::stoi(count_str);
                // Assign IDs to each output
                for (int i = 0; i < num_outputs; ++i) {
                    int id = next_id_++;
                    std::string indexed_name = base_name + "#" + std::to_string(i);
                    ssa_map_[indexed_name] = id;
                    op.outputs.push_back(id);
                }
            } catch (...) {
                // Parse error, treat as single output
                int id = next_id_++;
                ssa_map_[outputs_str] = id;
                op.outputs.push_back(id);
            }
        } else {
            // Single output case
            int id = next_id_++;
            ssa_map_[outputs_str] = id;
            op.outputs.push_back(id);
        }
        
        // Parse RHS: op_name operands : type
        std::string rhs = trim(line.substr(eq_pos + 1));
        
        // Extract operation name - find first space or '('
        size_t space_pos = rhs.find(' ');
        size_t paren_pos = rhs.find('(');
        size_t name_end = std::string::npos;
        
        if (space_pos != std::string::npos && paren_pos != std::string::npos) {
            name_end = std::min(space_pos, paren_pos);
        } else if (space_pos != std::string::npos) {
            name_end = space_pos;
        } else if (paren_pos != std::string::npos) {
            name_end = paren_pos;
        }
        
        if (name_end != std::string::npos) {
            op.op_name = rhs.substr(0, name_end);
        } else {
            op.op_name = rhs;
        }
        
        // Strip quotes if present (e.g. "stablehlo.reduce_window" -> stablehlo.reduce_window)
        if (op.op_name.length() >= 2 && op.op_name.front() == '"' && op.op_name.back() == '"') {
            op.op_name = op.op_name.substr(1, op.op_name.length() - 2);
        }
        
        // Special handling for call operations: call @function_name(args)
        // MLIR text may use either "call" or "func.call" prefix
        if (op.op_name == "call" || op.op_name == "func.call") {
            size_t at_pos = rhs.find('@');
            size_t call_paren = rhs.find('(', at_pos);
            if (at_pos != std::string::npos && call_paren != std::string::npos) {
                std::string func_name = rhs.substr(at_pos + 1, call_paren - at_pos - 1);
                op.op_name = "func.call";  // Match what executor expects
                op.attributes["callee"] = func_name;
                // Note: name_end needs adjustment for operand parsing
                name_end = call_paren;
            }
        }
        
        // Special handling for sdy.sharding_constraint - treat as passthrough
        if (op.op_name.find("sdy.") == 0) {
            op.op_name = "sdy_passthrough";  // We'll pass through the input unchanged
        }
        
        // Special handling for custom_call: stablehlo.custom_call @target(args)
        if (op.op_name == "stablehlo.custom_call" || op.op_name == "mhlo.custom_call") {
            size_t at_pos = rhs.find('@');
            if (at_pos != std::string::npos) {
                // Find end of target name (either '(' or space)
                size_t target_end = rhs.find_first_of("( ", at_pos + 1);
                if (target_end == std::string::npos) target_end = rhs.length();
                std::string target_name = rhs.substr(at_pos + 1, target_end - at_pos - 1);
                op.attributes["call_target_name"] = target_name;
            }
            
            // Extract mhlo.backend_config = {...} as a string attribute
            // The general parser can't handle dotted keys or nested dict values
            size_t bc_pos = rhs.find("mhlo.backend_config");
            if (bc_pos != std::string::npos) {
                size_t bc_brace = rhs.find('{', bc_pos);
                if (bc_brace != std::string::npos) {
                    // Find matching closing brace
                    int depth = 1;
                    size_t bc_end = bc_brace + 1;
                    while (bc_end < rhs.size() && depth > 0) {
                        if (rhs[bc_end] == '{') depth++;
                        if (rhs[bc_end] == '}') depth--;
                        bc_end++;
                    }
                    std::string bc_content = rhs.substr(bc_brace, bc_end - bc_brace);
                    op.attributes["mhlo.backend_config"] = bc_content;
                }
            }
        }
        
        // For iota ops, parse inline "dim = N" attribute
        // Format: stablehlo.iota dim = 0 : tensor<3x3xui64>
        if (op.op_name == "stablehlo.iota" || op.op_name == "mhlo.iota") {
            std::regex dim_regex(R"(dim\s*=\s*(\d+))");
            std::smatch match;
            if (std::regex_search(rhs, match, dim_regex)) {
                op.attributes["iota_dimension"] = match[1].str();
            }
        }
        
        // For compare ops, parse inline comparison direction
        // Format: stablehlo.compare  LT, %arg0, %1,  FLOAT : ...
        if (op.op_name == "stablehlo.compare" || op.op_name == "mhlo.compare") {
            // Look for the comparison direction (LT, LE, GT, GE, EQ, NE)
            std::regex cmp_regex(R"(\b(LT|LE|GT|GE|EQ|NE)\b)");
            std::smatch match;
            if (std::regex_search(rhs, match, cmp_regex)) {
                op.attributes["comparison_direction"] = match[1].str();
            }
        }
        
        // For convolution ops, parse dim_numbers and window attributes
        // Format: stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], ...}
        if (op.op_name == "stablehlo.convolution" || op.op_name == "mhlo.convolution") {
            // Parse dim_numbers: [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
            // Store the dim_numbers string so the conv handler can parse it
            size_t dn_pos = rhs.find("dim_numbers");
            if (dn_pos != std::string::npos) {
                size_t eq_pos = rhs.find('=', dn_pos);
                if (eq_pos != std::string::npos) {
                    // Find end of dim_numbers (before comma or brace that starts window/other attrs)
                    size_t dn_end = rhs.find(", window", eq_pos);
                    if (dn_end == std::string::npos) dn_end = rhs.find(", {", eq_pos);
                    if (dn_end == std::string::npos) dn_end = rhs.find(" {batch", eq_pos);
                    if (dn_end == std::string::npos) dn_end = rhs.length();
                    std::string dn_str = rhs.substr(eq_pos + 1, dn_end - eq_pos - 1);
                    // Trim whitespace
                    size_t start = dn_str.find_first_not_of(" ");
                    size_t end = dn_str.find_last_not_of(" ");
                    if (start != std::string::npos) dn_str = dn_str.substr(start, end - start + 1);
                    op.attributes["dim_numbers"] = dn_str;
                }
            }
            
            // Parse window attributes: stride, pad, lhs_dilate, rhs_dilate
            size_t window_pos = rhs.find("window = {");
            if (window_pos != std::string::npos) {
                size_t window_end = rhs.find("}", window_pos);
                if (window_end != std::string::npos) {
                    std::string window_str = rhs.substr(window_pos, window_end - window_pos + 1);
                    
                    // Parse stride = [1, 1]
                    std::regex stride_regex(R"(stride\s*=\s*\[([\d,\s]+)\])");
                    std::smatch stride_match;
                    if (std::regex_search(window_str, stride_match, stride_regex)) {
                        std::string stride_str = stride_match[1].str();
                        std::vector<int64_t> strides;
                        std::regex num_regex(R"(\d+)");
                        auto nums_begin = std::sregex_iterator(stride_str.begin(), stride_str.end(), num_regex);
                        auto nums_end = std::sregex_iterator();
                        for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                            strides.push_back(std::stoll(i->str()));
                        }
                        op.int_array_attrs["stride"] = strides;
                    }
                    
                    // Parse pad = [[0, 0], [0, 0]] (2D) or [[0, 0]] (1D)
                    // Try 2D first
                    std::regex pad_regex_2d(R"(pad\s*=\s*\[\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*\])");
                    std::smatch pad_match;
                    if (std::regex_search(window_str, pad_match, pad_regex_2d)) {
                        std::vector<int64_t> padding;
                        std::regex num_regex(R"(\d+)");
                        // First dimension
                        std::string pad0_str = pad_match[1].str();
                        auto it0 = std::sregex_iterator(pad0_str.begin(), pad0_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it0 != end) {
                            padding.push_back(std::stoll(it0->str()));
                            ++it0;
                        }
                        // Second dimension
                        std::string pad1_str = pad_match[2].str();
                        auto it1 = std::sregex_iterator(pad1_str.begin(), pad1_str.end(), num_regex);
                        while (it1 != end) {
                            padding.push_back(std::stoll(it1->str()));
                            ++it1;
                        }
                        if (!padding.empty()) op.int_array_attrs["padding"] = padding;
                    } else {
                        // Try 1D: pad = [[lo, hi]]
                        std::regex pad_regex_1d(R"(pad\s*=\s*\[\s*\[([^\]]*)\]\s*\])");
                        if (std::regex_search(window_str, pad_match, pad_regex_1d)) {
                            std::vector<int64_t> padding;
                            std::regex num_regex(R"(\d+)");
                            std::string pad0_str = pad_match[1].str();
                            auto it0 = std::sregex_iterator(pad0_str.begin(), pad0_str.end(), num_regex);
                            auto end = std::sregex_iterator();
                            while (it0 != end) {
                                padding.push_back(std::stoll(it0->str()));
                                ++it0;
                            }
                            if (!padding.empty()) op.int_array_attrs["padding"] = padding;
                        }
                    }
                    
                    // Parse lhs_dilate = [1, 1]
                    std::regex lhs_dil_regex(R"(lhs_dilate\s*=\s*\[([\d,\s]+)\])");
                    std::smatch lhs_dil_match;
                    if (std::regex_search(window_str, lhs_dil_match, lhs_dil_regex)) {
                        std::string dil_str = lhs_dil_match[1].str();
                        std::vector<int64_t> dilation;
                        std::regex num_regex(R"(\d+)");
                        auto it = std::sregex_iterator(dil_str.begin(), dil_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it != end) {
                            dilation.push_back(std::stoll(it->str()));
                            ++it;
                        }
                        if (!dilation.empty()) op.int_array_attrs["lhs_dilation"] = dilation;
                    }
                    
                    // Parse rhs_dilate = [1, 1]
                    std::regex rhs_dil_regex(R"(rhs_dilate\s*=\s*\[([\d,\s]+)\])");
                    std::smatch rhs_dil_match;
                    if (std::regex_search(window_str, rhs_dil_match, rhs_dil_regex)) {
                        std::string dil_str = rhs_dil_match[1].str();
                        std::vector<int64_t> dilation;
                        std::regex num_regex(R"(\d+)");
                        auto it = std::sregex_iterator(dil_str.begin(), dil_str.end(), num_regex);
                        auto end = std::sregex_iterator();
                        while (it != end) {
                            dilation.push_back(std::stoll(it->str()));
                            ++it;
                        }
                        if (!dilation.empty()) op.int_array_attrs["rhs_dilation"] = dilation;
                    }
                }
            }
        }
        
        // For dot_general ops
        if (op.op_name == "stablehlo.dot_general" || op.op_name == "mhlo.dot_general") {
             // Parse dot_dimension_numbers = {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]}
             // Also batching dimensions.
             
             size_t dims_start = rhs.find("dot_dimension_numbers = {");
             if (dims_start != std::string::npos) {
                 size_t dims_end = rhs.find("}", dims_start);
                 if (dims_end != std::string::npos) {
                     std::string dims_block = rhs.substr(dims_start, dims_end - dims_start);
                     
                     // Helper regex to extract arrays
                     auto parse_dims = [&](const std::string& key, const std::string& attr_name) {
                         std::string pattern = key + R"(\s*=\s*\[([\d,\s]*)\])";
                         std::regex r(pattern);
                         std::smatch m;
                         if (std::regex_search(dims_block, m, r)) {
                             std::string nums = m[1].str();
                             std::vector<int64_t> d;
                             if (!nums.empty()) {
                                 std::regex num_regex(R"(\d+)");
                                 auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                                 auto e = std::sregex_iterator();
                                 for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                             }
                             op.int_array_attrs[attr_name] = d;
                         }
                     };
                     
                     parse_dims("lhs_contracting_dimensions", "lhs_contracting");
                     parse_dims("rhs_contracting_dimensions", "rhs_contracting");
                     parse_dims("lhs_batching_dimensions", "lhs_batching");
                     parse_dims("rhs_batching_dimensions", "rhs_batching");
                 }
             }
             
             // Parse shorthand: contracting_dims = [0] x [0]
             std::regex contract_regex(R"(contracting_dims\s*=\s*\[([\d,\s]*)\]\s*x\s*\[([\d,\s]*)\])");
             std::smatch contract_match;
             if (std::regex_search(rhs, contract_match, contract_regex)) {
                 auto parse_list = [&](std::string nums) {
                     std::vector<int64_t> d;
                     std::regex num_regex(R"(\d+)");
                     auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto e = std::sregex_iterator();
                     for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                     return d;
                 };
                 op.int_array_attrs["lhs_contracting"] = parse_list(contract_match[1].str());
                 op.int_array_attrs["rhs_contracting"] = parse_list(contract_match[2].str());
             }

             // Parse shorthand: batching_dims = [0] x [0]
             std::regex batch_regex(R"(batching_dims\s*=\s*\[([\d,\s]*)\]\s*x\s*\[([\d,\s]*)\])");
             std::smatch batch_match;
             if (std::regex_search(rhs, batch_match, batch_regex)) {
                 auto parse_list = [&](std::string nums) {
                     std::vector<int64_t> d;
                     std::regex num_regex(R"(\d+)");
                     auto b = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto e = std::sregex_iterator();
                     for (auto i = b; i != e; ++i) d.push_back(std::stoll(i->str()));
                     return d;
                 };
                 op.int_array_attrs["lhs_batching"] = parse_list(batch_match[1].str());
                 op.int_array_attrs["rhs_batching"] = parse_list(batch_match[2].str());
             }
        }

        // For reduce ops
        if (op.op_name == "stablehlo.reduce" || op.op_name == "mhlo.reduce") {
             // Parse dimensions = [1]
             std::regex dims_regex(R"(dimensions\s*=\s*\[([\d,\s]+)\])");
             std::smatch dims_match;
             if (std::regex_search(rhs, dims_match, dims_regex)) {
                 std::string nums = dims_match[1].str();
                 std::vector<int64_t> dims;
                 std::regex num_regex(R"(\d+)");
                 auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                 auto nums_end = std::sregex_iterator();
                 for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                     dims.push_back(std::stoll(i->str()));
                 }
                 op.int_array_attrs["dimensions"] = dims;
             }
        }

        // For broadcast_in_dim
        if (op.op_name == "stablehlo.broadcast_in_dim" || op.op_name == "mhlo.broadcast_in_dim") {
            // Parse dims = [0, 1]
            std::regex dims_regex(R"(dims\s*=\s*\[([\d,\s]+)\])");
            std::smatch dims_match;
            if (std::regex_search(rhs, dims_match, dims_regex)) {
                std::string nums = dims_match[1].str();
                std::vector<int64_t> dims;
                std::regex num_regex(R"(\d+)");
                auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                auto nums_end = std::sregex_iterator();
                for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                    dims.push_back(std::stoll(i->str()));
                }
                op.int_array_attrs["broadcast_dimensions"] = dims;
            }
        }

        // For reduce_window ops (generic syntax)
        if (op.op_name == "stablehlo.reduce_window" || op.op_name == "mhlo.reduce_window") {
            // Check for generic attributes block <{...}>
            size_t attr_start = rhs.find("<{");
            size_t attr_end = rhs.find("}>");
            if (attr_start != std::string::npos && attr_end != std::string::npos) {
                 std::string attr_block = rhs.substr(attr_start, attr_end - attr_start);
                 
                 // Parse window_dimensions = array<i64: 1, 2, 2, 1>
                 std::regex dim_regex(R"(window_dimensions\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch dim_match;
                 if (std::regex_search(attr_block, dim_match, dim_regex)) {
                     std::string nums = dim_match[1].str();
                     std::vector<int64_t> dims;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         dims.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_dimensions"] = dims;
                 }
                 
                 // Parse window_strides = array<i64: 1, 2, 2, 1>
                 std::regex stride_regex(R"(window_strides\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch stride_match;
                 if (std::regex_search(attr_block, stride_match, stride_regex)) {
                     std::string nums = stride_match[1].str();
                     std::vector<int64_t> strides;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         strides.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_strides"] = strides;
                 }
                 
                 // Parse padding = dense<[[3, 0]]> : tensor<1x2xi64>
                 std::regex pad_regex(R"(padding\s*=\s*dense<)");
                 std::smatch pad_match;
                 if (std::regex_search(attr_block, pad_match, pad_regex)) {
                     size_t dense_start = attr_block.find("dense<", pad_match.position());
                     size_t dense_end = attr_block.find(">", dense_start + 6);
                     if (dense_start != std::string::npos && dense_end != std::string::npos) {
                         std::string dense_content = attr_block.substr(dense_start + 6, dense_end - dense_start - 6);
                         std::vector<int64_t> padding;
                         std::regex num_regex(R"(-?\d+)");
                         auto nums_begin = std::sregex_iterator(dense_content.begin(), dense_content.end(), num_regex);
                         auto nums_end = std::sregex_iterator();
                         for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                             padding.push_back(std::stoll(i->str()));
                         }
                         if (!padding.empty()) op.int_array_attrs["padding"] = padding;
                     }
                 }
            }
        }

        // For select_and_scatter ops (pool gradient) - parse same window attributes
        if (op.op_name == "stablehlo.select_and_scatter" || op.op_name == "mhlo.select_and_scatter") {
            size_t attr_start = rhs.find("<{");
            size_t attr_end = rhs.find("}>");
            if (attr_start != std::string::npos && attr_end != std::string::npos) {
                 std::string attr_block = rhs.substr(attr_start, attr_end - attr_start);
                 
                 // Parse window_dimensions = array<i64: 1, 2, 2, 1>
                 std::regex dim_regex(R"(window_dimensions\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch dim_match;
                 if (std::regex_search(attr_block, dim_match, dim_regex)) {
                     std::string nums = dim_match[1].str();
                     std::vector<int64_t> dims;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         dims.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_dimensions"] = dims;
                 }
                 
                 // Parse window_strides = array<i64: 1, 2, 2, 1>
                 std::regex stride_regex(R"(window_strides\s*=\s*array<i64:\s*([\d,\s]+)>)");
                 std::smatch stride_match;
                 if (std::regex_search(attr_block, stride_match, stride_regex)) {
                     std::string nums = stride_match[1].str();
                     std::vector<int64_t> strides;
                     std::regex num_regex(R"(\d+)");
                     auto nums_begin = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                     auto nums_end = std::sregex_iterator();
                     for (std::sregex_iterator i = nums_begin; i != nums_end; ++i) {
                         strides.push_back(std::stoll(i->str()));
                     }
                     op.int_array_attrs["window_strides"] = strides;
                 }
            }
        }

        // For pad ops, parse low/high/interior padding from custom assembly format
        // Format: stablehlo.pad %input, %val, low = [0, 0], high = [1, 0], interior = [1, 0] : ...
        // Also generic format: <{edge_padding_low = array<i64: 0, 0>, ...}>
        if (op.op_name == "stablehlo.pad" || op.op_name == "mhlo.pad") {
            auto parse_bracket_array = [&](const std::string& text, const std::string& key) -> std::vector<int64_t> {
                std::vector<int64_t> result;
                std::string pattern = key + R"(\s*=\s*\[([\d,\s-]*)\])";
                std::regex r(pattern);
                std::smatch m;
                if (std::regex_search(text, m, r)) {
                    std::string nums = m[1].str();
                    std::regex num_regex(R"(-?\d+)");
                    auto it = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                    auto end = std::sregex_iterator();
                    for (; it != end; ++it) {
                        result.push_back(std::stoll(it->str()));
                    }
                }
                return result;
            };

            // Custom assembly format: low = [...], high = [...], interior = [...]
            auto low = parse_bracket_array(rhs, "low");
            if (!low.empty()) op.int_array_attrs["edge_padding_low"] = low;
            auto high = parse_bracket_array(rhs, "high");
            if (!high.empty()) op.int_array_attrs["edge_padding_high"] = high;
            auto interior = parse_bracket_array(rhs, "interior");
            if (!interior.empty()) op.int_array_attrs["interior_padding"] = interior;

            // Generic format: <{edge_padding_low = array<i64: 0, 0>, ...}>
            size_t attr_start = rhs.find("<{");
            size_t attr_end = rhs.find("}>");
            if (attr_start != std::string::npos && attr_end != std::string::npos) {
                std::string attr_block = rhs.substr(attr_start, attr_end - attr_start);
                auto parse_array_attr = [&](const std::string& key) -> std::vector<int64_t> {
                    std::vector<int64_t> result;
                    std::string pattern = key + R"(\s*=\s*array<i64:\s*([\d,\s-]+)>)";
                    std::regex r(pattern);
                    std::smatch m;
                    if (std::regex_search(attr_block, m, r)) {
                        std::string nums = m[1].str();
                        std::regex num_regex(R"(-?\d+)");
                        auto it = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                        auto end = std::sregex_iterator();
                        for (; it != end; ++it) result.push_back(std::stoll(it->str()));
                    }
                    return result;
                };
                if (!op.int_array_attrs.count("edge_padding_low")) {
                    auto v = parse_array_attr("edge_padding_low");
                    if (!v.empty()) op.int_array_attrs["edge_padding_low"] = v;
                }
                if (!op.int_array_attrs.count("edge_padding_high")) {
                    auto v = parse_array_attr("edge_padding_high");
                    if (!v.empty()) op.int_array_attrs["edge_padding_high"] = v;
                }
                if (!op.int_array_attrs.count("interior_padding")) {
                    auto v = parse_array_attr("interior_padding");
                    if (!v.empty()) op.int_array_attrs["interior_padding"] = v;
                }
            }
        }

        // For gather ops, parse dimension_numbers and slice_sizes
        // Generic format: <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], 
        //   collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>,
        //   slice_sizes = array<i64: 1, 1, 128, 256>}>
        if (op.op_name == "stablehlo.gather" || op.op_name == "mhlo.gather") {
            auto parse_bracket_ints = [](const std::string& text, const std::string& attr_name) -> std::vector<int64_t> {
                std::vector<int64_t> result;
                size_t pos = text.find(attr_name + " = [");
                if (pos == std::string::npos) pos = text.find(attr_name + "= [");
                if (pos == std::string::npos) pos = text.find(attr_name + " =[");
                if (pos != std::string::npos) {
                    size_t bracket_start = text.find('[', pos);
                    size_t bracket_end = text.find(']', bracket_start);
                    if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                        std::string nums = text.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                        std::regex num_regex(R"(-?\d+)");
                        auto it = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                        auto end = std::sregex_iterator();
                        for (; it != end; ++it) {
                            result.push_back(std::stoll(it->str()));
                        }
                    }
                }
                return result;
            };

            auto offset = parse_bracket_ints(rhs, "offset_dims");
            if (!offset.empty()) op.int_array_attrs["offset_dims"] = offset;

            auto collapsed = parse_bracket_ints(rhs, "collapsed_slice_dims");
            if (!collapsed.empty()) op.int_array_attrs["collapsed_slice_dims"] = collapsed;

            auto start_map = parse_bracket_ints(rhs, "start_index_map");
            if (!start_map.empty()) op.int_array_attrs["start_index_map"] = start_map;

            // index_vector_dim = N (scalar inside the #stablehlo.gather<...>)
            std::regex ivd_regex(R"(index_vector_dim\s*=\s*(\d+))");
            std::smatch ivd_match;
            if (std::regex_search(rhs, ivd_match, ivd_regex)) {
                op.int_attrs["index_vector_dim"] = std::stoll(ivd_match[1].str());
            }

            // slice_sizes = array<i64: 1, 1, 128, 256>
            std::regex ss_regex(R"(slice_sizes\s*=\s*array<i64:\s*([\d,\s]+)>)");
            std::smatch ss_match;
            if (std::regex_search(rhs, ss_match, ss_regex)) {
                std::string nums = ss_match[1].str();
                std::vector<int64_t> sizes;
                std::regex num_regex(R"(\d+)");
                auto it = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                auto end = std::sregex_iterator();
                for (; it != end; ++it) {
                    sizes.push_back(std::stoll(it->str()));
                }
                op.int_array_attrs["slice_sizes"] = sizes;
            }
        }

        // For concatenate ops, parse inline "dim" / "dimension" attribute
        // Custom assembly format: stablehlo.concatenate %a, %b, dim = 1 : ...
        // Generic format: <{dimension = 1 : i64}>
        if (op.op_name == "stablehlo.concatenate" || op.op_name == "mhlo.concatenate") {
            // Match both "dim = N" and "dimension = N" (with optional type suffix)
            std::regex dim_regex(R"(\bdim(?:ension)?\s*=\s*(\d+))");
            std::smatch dim_match;
            if (std::regex_search(rhs, dim_match, dim_regex)) {
                op.int_attrs["dimension"] = std::stoll(dim_match[1].str());
            }
        }
        
        // For FFT ops, parse inline "type = FFT/IFFT/RFFT/IRFFT, length = [N]" attributes
        // Custom assembly format: stablehlo.fft %arg0, type =  IFFT, length = [3]
        if (op.op_name == "stablehlo.fft" || op.op_name == "mhlo.fft") {
            // Extract FFT type (FFT, IFFT, RFFT, IRFFT)
            std::regex type_regex(R"(type\s*=\s*(I?R?FFT))");
            std::smatch type_match;
            if (std::regex_search(rhs, type_match, type_regex)) {
                op.attributes["fft_type"] = type_match[1].str();
            }
            // Extract length array
            std::regex len_regex(R"(length\s*=\s*\[([^\]]*)\])");
            std::smatch len_match;
            if (std::regex_search(rhs, len_match, len_regex)) {
                std::string nums = len_match[1].str();
                std::vector<int64_t> lengths;
                std::regex num_regex(R"(\d+)");
                auto it = std::sregex_iterator(nums.begin(), nums.end(), num_regex);
                auto end = std::sregex_iterator();
                for (; it != end; ++it) {
                    lengths.push_back(std::stoll(it->str()));
                }
                op.int_array_attrs["fft_length"] = lengths;
            }
        }
        // For reduce ops, extract the reduce function from "applies stablehlo.add"
        if (op.op_name == "stablehlo.reduce" || op.op_name == "mhlo.reduce") {
            size_t applies_pos = rhs.find("applies ");
            if (applies_pos != std::string::npos) {
                size_t fn_start = applies_pos + 8;  // After "applies "
                size_t fn_end = rhs.find(' ', fn_start);
                if (fn_end == std::string::npos) fn_end = rhs.length();
                std::string reduce_fn = rhs.substr(fn_start, fn_end - fn_start);
                // Map reduce function to reduce_type
                if (reduce_fn.find("add") != std::string::npos) {
                    op.attributes["reduce_type"] = "sum";
                } else if (reduce_fn.find("maximum") != std::string::npos || reduce_fn.find("max") != std::string::npos) {
                    op.attributes["reduce_type"] = "max";
                } else if (reduce_fn.find("minimum") != std::string::npos || reduce_fn.find("min") != std::string::npos) {
                    op.attributes["reduce_type"] = "min";
                } else if (reduce_fn.find("multiply") != std::string::npos || reduce_fn.find("mul") != std::string::npos) {
                    op.attributes["reduce_type"] = "prod";
                } else if (reduce_fn.find("or") != std::string::npos) {
                    op.attributes["reduce_type"] = "or";
                } else if (reduce_fn.find("and") != std::string::npos) {
                    op.attributes["reduce_type"] = "and";
                } else {
                    op.attributes["reduce_type"] = "sum";  // Default
                }
            }
            
            // Extract dimensions from "dimensions = [0]" or "across dimensions = [0]"
            size_t dim_pos = rhs.find("dimensions = [");
            if (dim_pos != std::string::npos) {
                size_t bracket_start = rhs.find('[', dim_pos);
                size_t bracket_end = rhs.find(']', bracket_start);
                if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                    std::string dim_str = rhs.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                    std::vector<int64_t> dims;
                    std::regex num_regex(R"(-?\d+)");
                    std::sregex_iterator dim_iter(dim_str.begin(), dim_str.end(), num_regex);
                    std::sregex_iterator dim_end;
                    while (dim_iter != dim_end) {
                        dims.push_back(std::stoll((*dim_iter)[0].str()));
                        ++dim_iter;
                    }
                    op.int_array_attrs["dimensions"] = dims;
                }
            }
        }
        
        // For slice ops, parse inline [start:limit, ...] syntax
        // Format: stablehlo.slice %arg0 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        //         stablehlo.slice %arg0 [0:2, 1:3] : (tensor<4x5xf32>) -> tensor<2x2xf32>
        if (op.op_name == "stablehlo.slice" || op.op_name == "mhlo.slice") {
            size_t bracket_start = rhs.find('[');
            size_t bracket_end = rhs.find(']');
            if (bracket_start != std::string::npos && bracket_end != std::string::npos && bracket_end > bracket_start) {
                std::string slice_spec = rhs.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                // Parse each dimension's start:limit[:stride]
                std::vector<int64_t> starts, limits, strides;
                
                // Split by comma
                size_t pos = 0;
                while (pos < slice_spec.length()) {
                    // Find next comma or end
                    size_t next_comma = slice_spec.find(',', pos);
                    std::string dim_spec = (next_comma == std::string::npos) 
                        ? slice_spec.substr(pos) 
                        : slice_spec.substr(pos, next_comma - pos);
                    
                    // Parse start:limit[:stride]
                    size_t colon1 = dim_spec.find(':');
                    if (colon1 != std::string::npos) {
                        std::string start_str = trim(dim_spec.substr(0, colon1));
                        std::string rest = dim_spec.substr(colon1 + 1);
                        
                        size_t colon2 = rest.find(':');
                        std::string limit_str = (colon2 == std::string::npos) 
                            ? trim(rest) 
                            : trim(rest.substr(0, colon2));
                        std::string stride_str = (colon2 == std::string::npos) 
                            ? "1" 
                            : trim(rest.substr(colon2 + 1));
                        
                        try {
                            starts.push_back(std::stoll(start_str));
                            limits.push_back(std::stoll(limit_str));
                            strides.push_back(stride_str.empty() ? 1 : std::stoll(stride_str));
                        } catch (...) {
                            // Parse error, skip
                        }
                    }
                    
                    if (next_comma == std::string::npos) break;
                    pos = next_comma + 1;
                }
                
                if (!starts.empty()) {
                    op.int_array_attrs["start_indices"] = starts;
                    op.int_array_attrs["limit_indices"] = limits;
                    op.int_array_attrs["strides"] = strides;
                }
            }
        }
        
        // For constant ops, parse dense<...> values
        // Format: stablehlo.constant dense<32> : tensor<i32>
        //         stablehlo.constant dense<[1, 2, 3]> : tensor<3xi32>
        //         stablehlo.constant dense<1.0> : tensor<f32>
        if (op.op_name == "stablehlo.constant" || op.op_name == "mhlo.constant") {
            size_t dense_pos = rhs.find("dense<");
            if (dense_pos != std::string::npos) {
                size_t value_start = dense_pos + 6;  // After "dense<"
                size_t value_end = rhs.find('>', value_start);
                if (value_end != std::string::npos) {
                    std::string value_str = rhs.substr(value_start, value_end - value_start);
                    
                    // Check if it's a boolean literal (true/false)
                    bool is_bool = (value_str.find("true") != std::string::npos || 
                                    value_str.find("false") != std::string::npos);
                    
                    // Check if it's a hex float literal (e.g. 0xFF800000 = -inf)
                    // MLIR uses this for special IEEE 754 values
                    bool is_hex = (value_str.find("0x") != std::string::npos ||
                                   value_str.find("0X") != std::string::npos);
                    
                    // Check if it's a float (contains '.' or 'e')
                    bool is_float = (value_str.find('.') != std::string::npos || 
                                     value_str.find('e') != std::string::npos ||
                                     value_str.find('E') != std::string::npos);

                    
                    if (is_bool) {
                        // Parse as boolean array (store as 0/1 in int_array)
                        std::vector<int64_t> int_vals;
                        std::regex bool_regex(R"(\b(true|false)\b)");
                        std::sregex_iterator biter(value_str.begin(), value_str.end(), bool_regex);
                        std::sregex_iterator bend;
                        while (biter != bend) {
                            std::string match = (*biter)[0].str();
                            int_vals.push_back(match == "true" ? 1 : 0);
                            ++biter;
                        }
                        if (!int_vals.empty()) {
                            op.int_array_attrs["value"] = int_vals;
                        }
                    } else if (is_hex) {
                        // Parse hex float literals (e.g. 0xFF800000 = -inf in IEEE 754)
                        // MLIR uses these for special values like -inf, inf, NaN
                        // Handle mixed hex+float constants like [1.0, 0x7FC00000, 2.0]
                        std::vector<float> float_vals;
                        if (is_float) {
                            // Mixed format: use combined regex matching both hex and decimal floats
                            // Regex matches hex values OR decimal float values, preserving order
                            std::regex mixed_regex(R"(0[xX][0-9a-fA-F]+|-?[\d.]+(?:[eE][+-]?\d+)?)");
                            std::sregex_iterator miter(value_str.begin(), value_str.end(), mixed_regex);
                            std::sregex_iterator mend;
                            while (miter != mend) {
                                std::string token = (*miter)[0].str();
                                try {
                                    if (token.find("0x") != std::string::npos || token.find("0X") != std::string::npos) {
                                        uint32_t bits = static_cast<uint32_t>(std::stoul(token, nullptr, 16));
                                        float f;
                                        std::memcpy(&f, &bits, sizeof(f));
                                        float_vals.push_back(f);
                                    } else {
                                        float_vals.push_back(std::stof(token));
                                    }
                                } catch (...) {}
                                ++miter;
                            }
                        } else {
                            // Pure hex format
                            std::regex hex_regex(R"(0[xX][0-9a-fA-F]+)");
                            std::sregex_iterator hiter(value_str.begin(), value_str.end(), hex_regex);
                            std::sregex_iterator hend;
                            while (hiter != hend) {
                                try {
                                    uint32_t bits = static_cast<uint32_t>(std::stoul((*hiter)[0].str(), nullptr, 16));
                                    float f;
                                    std::memcpy(&f, &bits, sizeof(f));
                                    float_vals.push_back(f);
                                } catch (...) {}
                                ++hiter;
                            }
                        }
                        if (!float_vals.empty()) {
                            op.float_array_attrs["value"] = float_vals;
                        }
                    } else if (is_float) {

                        // Parse as float array
                        std::vector<float> float_vals;
                        std::regex float_regex(R"(-?[\d.]+(?:[eE][+-]?\d+)?)");
                        std::sregex_iterator fiter(value_str.begin(), value_str.end(), float_regex);
                        std::sregex_iterator fend;
                        while (fiter != fend) {
                            try {
                                float_vals.push_back(std::stof((*fiter)[0].str()));
                            } catch (...) {}
                            ++fiter;
                        }
                        if (!float_vals.empty()) {
                            op.float_array_attrs["value"] = float_vals;
                        }
                    } else {
                        // Parse as int array
                        std::vector<int64_t> int_vals;
                        std::regex int_regex(R"(-?\d+)");
                        std::sregex_iterator iiter(value_str.begin(), value_str.end(), int_regex);
                        std::sregex_iterator iend;
                        while (iiter != iend) {
                            try {
                                int_vals.push_back(std::stoll((*iiter)[0].str()));
                            } catch (...) {}
                            ++iiter;
                        }
                        if (!int_vals.empty()) {
                            op.int_array_attrs["value"] = int_vals;
                        }
                    }
                }
            }
        }
        
        // Extract operands and type info
        size_t colon_pos = rhs.rfind(':');
        std::string operands_str;
        std::string type_str;
        
        if (colon_pos != std::string::npos && name_end != std::string::npos) {
            operands_str = rhs.substr(name_end + 1, colon_pos - name_end - 1);
            type_str = trim(rhs.substr(colon_pos + 1));
        } else if (name_end != std::string::npos) {
            operands_str = rhs.substr(name_end + 1);
        }
        
        // Parse operands (space or comma separated %names)
        std::regex operand_regex(R"(%[\w#]+)");
        std::sregex_iterator iter(operands_str.begin(), operands_str.end(), operand_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string operand = (*iter)[0].str();
            if (ssa_map_.count(operand)) {
                op.inputs.push_back(ssa_map_[operand]);
            }
            ++iter;
        }
        
        // Parse output type
        if (!type_str.empty()) {
            // Handle arrow type format: (input) -> output
            // We want the OUTPUT type (after ->)
            std::string output_type_str = type_str;
            size_t arrow_pos = type_str.find("->");
            if (arrow_pos != std::string::npos) {
                output_type_str = trim(type_str.substr(arrow_pos + 2));
            }
            
            // Handle tuple types: (tensor<...>, tensor<...>) for multiple outputs
            // Also handle simple type: tensor<...>
            std::regex tensor_regex(R"(tensor<([^>]+)>)");
            std::sregex_iterator type_iter(output_type_str.begin(), output_type_str.end(), tensor_regex);
            std::sregex_iterator type_end;
            
            while (type_iter != type_end) {
                std::string shape_dtype = (*type_iter)[1].str();
                op.output_shapes.push_back(parseShape(shape_dtype));
                op.output_dtypes.push_back(parseDtype(shape_dtype));
                ++type_iter;
            }
        }
        
        // Parse attributes in {...}
        // Skip for gather ops — their #stablehlo.gather<...> syntax with nested brackets
        // confuses the generic comma-based parser (e.g., [0, 1] splits on commas)
        if (op.op_name != "stablehlo.gather" && op.op_name != "mhlo.gather") {
            size_t attr_start = rhs.find('{');
            size_t attr_end = rhs.rfind('}');
            if (attr_start != std::string::npos && attr_end != std::string::npos) {
                std::string attrs = rhs.substr(attr_start + 1, attr_end - attr_start - 1);
                parseAttributes(attrs, op);
            }
        }
        
        // Generic custom-assembly attribute parser
        // Catches patterns like: sizes = [3], dims = [0, 1], dim = 0
        // that appear in MLIR custom assembly format (outside of <{...}> blocks)
        // Only adds attributes not already parsed by op-specific parsers above.
        // IMPORTANT: Only scan the portion BEFORE the type separator " : " to avoid
        // matching numbers/brackets in type annotations like (tensor<1x1x5xf32>).
        {
            // Extract the attribute region (before type annotation)
            std::string attr_region = rhs;
            size_t type_sep = rhs.find(" : ");
            if (type_sep != std::string::npos) {
                attr_region = rhs.substr(0, type_sep);
            }
            
            // Match: identifier = [int, int, ...] (bracket arrays)
            std::regex bracket_attr_regex(R"((\w+)\s*=\s*\[([^\]]*)\])");
            auto bracket_begin = std::sregex_iterator(attr_region.begin(), attr_region.end(), bracket_attr_regex);
            auto bracket_end_it = std::sregex_iterator();
            for (auto it = bracket_begin; it != bracket_end_it; ++it) {
                std::string key = (*it)[1].str();
                std::string values_str = (*it)[2].str();
                // Skip if already parsed, or if key is a type annotation
                if (op.int_array_attrs.count(key) || key == "tensor" || key == "array") continue;
                // Parse comma-separated integers
                std::vector<int64_t> values;
                std::regex num_regex(R"(-?\d+)");
                auto num_begin = std::sregex_iterator(values_str.begin(), values_str.end(), num_regex);
                auto num_end_it = std::sregex_iterator();
                for (auto nit = num_begin; nit != num_end_it; ++nit) {
                    try { values.push_back(std::stoll((*nit)[0].str())); } catch (...) {}
                }
                if (!values.empty()) {
                    op.int_array_attrs[key] = values;
                }
            }
            
            // Match: key = integer (single scalar, e.g. "dim = 0" for iota)
            std::regex scalar_attr_regex(R"(,\s*(\w+)\s*=\s*(-?\d+)(?:\s|$|:))");
            auto scalar_begin = std::sregex_iterator(attr_region.begin(), attr_region.end(), scalar_attr_regex);
            auto scalar_end_it = std::sregex_iterator();
            for (auto it = scalar_begin; it != scalar_end_it; ++it) {
                std::string key = (*it)[1].str();
                if (op.int_array_attrs.count(key)) continue;
                if (key == "tensor" || key == "i32" || key == "f32" || key == "i64") continue;
                try {
                    op.int_array_attrs[key] = {std::stoll((*it)[2].str())};
                } catch (...) {}
            }
            
            // Also match standalone "key = N" for ops like iota (no leading comma)
            if (op.op_name.find("iota") != std::string::npos) {
                std::regex dim_regex(R"(dim\s*=\s*(-?\d+))");
                std::smatch dim_match;
                if (std::regex_search(attr_region, dim_match, dim_regex) && !op.int_array_attrs.count("dim")) {
                    op.int_array_attrs["dim"] = {std::stoll(dim_match[1].str())};
                }
            }
        }
        
        graph.nodes.push_back(op);
    }
    
    void parseReturn(const std::string& line, MLXGraph& graph) {
        // Parse: func.return %0 : tensor<2x3xf32>
        std::regex ret_regex(R"(%[\w#]+)");
        std::sregex_iterator iter(line.begin(), line.end(), ret_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string name = (*iter)[0].str();
            if (ssa_map_.count(name)) {
                graph.output_ids.push_back(ssa_map_[name]);
            }
            ++iter;
        }
    }
    
    std::vector<int> parseShape(const std::string& shape_dtype) {
        std::vector<int> shape;
        // Parse "2x3xf32" -> [2, 3]
        std::regex dim_regex(R"((\d+)x)");
        std::sregex_iterator iter(shape_dtype.begin(), shape_dtype.end(), dim_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            shape.push_back(std::stoi((*iter)[1].str()));
            ++iter;
        }
        
        // Handle scalar case (no dimensions, just dtype like "f32")
        if (shape.empty() && !shape_dtype.empty()) {
            // Check if it's purely a dtype (no dimensions)
            if (shape_dtype.find('x') == std::string::npos) {
                // Scalar - empty shape is correct
            }
        }
        
        return shape;
    }
    
    std::string parseDtype(const std::string& shape_dtype) {
        // Extract dtype from end of "2x3xf32" or just "f32"
        size_t last_x = shape_dtype.rfind('x');
        if (last_x != std::string::npos) {
            return shape_dtype.substr(last_x + 1);
        }
        return shape_dtype;  // Pure dtype like "f32"
    }
    
    void parseAttributes(const std::string& attrs, MLXOp& op) {
        // Parse: key = value, key2 = value2
        // Simple regex-based parsing
        std::regex attr_regex(R"((\w+)\s*=\s*([^,}]+))");
        std::sregex_iterator iter(attrs.begin(), attrs.end(), attr_regex);
        std::sregex_iterator end;
        
        while (iter != end) {
            std::string key = (*iter)[1].str();
            std::string value = trim((*iter)[2].str());
            
            // Try to parse as integer (handles both "1" and "1 : i64" / "1 : si64")
            try {
                size_t pos;
                int64_t int_val = std::stoll(value, &pos);
                // Accept if we parsed at least one digit and remaining is type suffix or empty
                if (pos > 0 && (pos == value.length() || value[pos] == ' ' || value[pos] == ':')) {
                    op.int_attrs[key] = int_val;
                }
            } catch (...) {}
            
            // Try to parse as int array [1, 2, 3] or dense<[1, 2]>
            if (value[0] == '[' || value.find("dense<") != std::string::npos) {
                std::vector<int64_t> arr;
                std::regex num_regex(R"(-?\d+)");
                std::sregex_iterator num_iter(value.begin(), value.end(), num_regex);
                std::sregex_iterator num_end;
                while (num_iter != num_end) {
                    arr.push_back(std::stoll((*num_iter)[0].str()));
                    ++num_iter;
                }
                if (!arr.empty()) {
                    op.int_array_attrs[key] = arr;
                }
            }
            
            // Store as string attr always
            op.attributes[key] = value;
            
            ++iter;
        }
    }
    
    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }
};

// Main entry point
inline bool ParseMLIRText(const std::string& text, MLXGraph& graph) {
    MLIRParser parser;
    return parser.parse(text, graph);
}

// Extended entry point that also parses private functions
inline bool ParseMLIRText(const std::string& text, MLXGraph& graph, 
                          std::map<std::string, std::shared_ptr<MLXGraph>>& functions) {
    MLIRParser parser;
    return parser.parseAll(text, graph, functions);
}

} // namespace mlx_parser

#endif // MLX_MLIR_PARSER_H
