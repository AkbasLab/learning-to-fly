#!/usr/bin/env python3
"""
Checkpoint Conversion Pipeline for Learning-to-Fly Firmware

This script converts rl_tools MLP checkpoint files (from PPO, SAC, etc.) into the
Sequential module format expected by the Crazyflie firmware. This allows any
policy type to be deployed without modifying the firmware adapter.

Architecture: 146 inputs → 64 hidden → 64 hidden → 4 outputs (FAST_TANH activation)

Usage:
    python convert_checkpoint_to_firmware.py <input_checkpoint.h> <output_checkpoint.h> [--name "Model Name"]
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def extract_memory_arrays(content: str) -> dict:
    """
    Extract all memory arrays from an MLP checkpoint file.
    Returns a dict mapping layer names to (weights_bytes, biases_bytes) tuples.
    """
    layers = {}
    
    # Pattern to match memory arrays: alignas(float) const unsigned char memory[] = {...};
    # We need to identify which layer each array belongs to
    
    # Find input_layer weights and biases
    input_layer_match = re.search(
        r'namespace input_layer\s*\{.*?namespace weights\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    input_layer_biases_match = re.search(
        r'namespace input_layer\s*\{.*?namespace biases\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    
    if input_layer_match and input_layer_biases_match:
        layers['layer_0'] = (input_layer_match.group(1).strip(), input_layer_biases_match.group(1).strip())
    
    # Find hidden_layer_0 weights and biases
    hidden_layer_match = re.search(
        r'namespace hidden_layer_0\s*\{.*?namespace weights\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    hidden_layer_biases_match = re.search(
        r'namespace hidden_layer_0\s*\{.*?namespace biases\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    
    if hidden_layer_match and hidden_layer_biases_match:
        layers['layer_1'] = (hidden_layer_match.group(1).strip(), hidden_layer_biases_match.group(1).strip())
    
    # Find output_layer weights and biases
    output_layer_match = re.search(
        r'namespace output_layer\s*\{.*?namespace weights\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    output_layer_biases_match = re.search(
        r'namespace output_layer\s*\{.*?namespace biases\s*\{.*?namespace parameters_memory\s*\{.*?'
        r'alignas\(float\)\s*const\s*unsigned\s*char\s*memory\[\]\s*=\s*\{([^}]+)\}',
        content, re.DOTALL
    )
    
    if output_layer_match and output_layer_biases_match:
        layers['layer_2'] = (output_layer_match.group(1).strip(), output_layer_biases_match.group(1).strip())
    
    return layers


def generate_sequential_checkpoint(layers: dict, model_name: str = "converted_policy") -> str:
    """
    Generate a Sequential module checkpoint file from extracted layer data.
    This matches the exact format expected by the learning-to-fly firmware.
    """
    
    # Layer dimensions for 146->64->64->4 network
    layer_specs = [
        ('layer_0', 146, 64, 'Input'),
        ('layer_1', 64, 64, 'Normal'),
        ('layer_2', 64, 4, 'Output'),
    ]
    
    output = []
    
    # Header
    output.append('''// Auto-generated checkpoint file for Learning-to-Fly firmware
// Converted from MLP format to Sequential format
// Model: {model_name}

#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn/layers/dense/layer.h>
#include <rl_tools/nn/parameters/parameters.h>

namespace rl_tools::checkpoint::actor {{
'''.format(model_name=model_name))
    
    # Generate each layer
    for layer_name, input_dim, output_dim, group_type in layer_specs:
        if layer_name not in layers:
            raise ValueError(f"Missing layer: {layer_name}")
        
        weights_data, biases_data = layers[layer_name]
        
        output.append(f'''    namespace {layer_name} {{
        namespace weights {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{weights_data}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned int, {output_dim}, {input_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}}; 
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Weights>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        namespace biases {{
            namespace parameters_memory {{
                static_assert(sizeof(unsigned char) == 1);
                alignas(float) const unsigned char memory[] = {{{biases_data}}};
                using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned int, 1, {output_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
                using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
                const CONTAINER_TYPE container = {{(float*)memory}}; 
            }}
            using PARAMETER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::spec<parameters_memory::CONTAINER_TYPE, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::categories::Biases>;
            const RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain::instance<PARAMETER_SPEC> parameters = {{parameters_memory::container}};
        }}
        using SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Specification<float, unsigned int, {input_dim}, {output_dim}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::activation_functions::ActivationFunction::FAST_TANH, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::Plain, 1, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::parameters::groups::{group_type}, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamicTag, true, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned int, 1>>;
        using TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::nn::layers::dense::Layer<SPEC>;
        const TYPE layer = {{weights::parameters, biases::parameters}};
    }}
''')
    
    # Model definition using Sequential Module
    output.append('''    namespace model_definition {
        using namespace rl_tools::nn_models::sequential::interface;
        using MODEL = Module<layer_0::TYPE, Module<layer_1::TYPE, Module<layer_2::TYPE>>>;
    }
    using MODEL = model_definition::MODEL;
    const MODEL model = {layer_0::layer, {layer_1::layer, {layer_2::layer}}};
}
''')
    
    # Observation namespace - must use MatrixDynamic format like the working checkpoint
    # Generate zero bytes for 146 floats (146 * 4 = 584 bytes)
    obs_bytes = ', '.join(['0'] * (146 * 4))
    
    output.append(f'''#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::observation {{
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {{{obs_bytes}}};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 146, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {{(float*)memory}}; 
}}

#include <rl_tools/containers.h>
namespace rl_tools::checkpoint::action {{
    static_assert(sizeof(unsigned char) == 1);
    alignas(float) const unsigned char memory[] = {{0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63}};
    using CONTAINER_SPEC = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::Specification<float, unsigned long, 1, 4, RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::matrix::layouts::RowMajorAlignment<unsigned long, 1>>;
    using CONTAINER_TYPE = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::MatrixDynamic<CONTAINER_SPEC>;
    const CONTAINER_TYPE container = {{(float*)memory}}; 
}}

namespace rl_tools::checkpoint::meta {{
    char name[] = "{model_name}";
    char commit_hash[] = "ppo_converted";
}}
''')
    
    return ''.join(output)


def convert_checkpoint(input_path: str, output_path: str, model_name: Optional[str] = None) -> bool:
    """
    Convert an MLP checkpoint to Sequential format.
    
    Args:
        input_path: Path to the input MLP checkpoint file
        output_path: Path for the output Sequential checkpoint file
        model_name: Optional model name for metadata
    
    Returns:
        True if conversion successful, False otherwise
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Read input file
    content = input_file.read_text()
    
    # Extract layer data
    print(f"Extracting layers from: {input_path}")
    layers = extract_memory_arrays(content)
    
    if len(layers) != 3:
        print(f"Error: Expected 3 layers, found {len(layers)}: {list(layers.keys())}")
        return False
    
    print(f"  Found layers: {list(layers.keys())}")
    
    # Generate model name from input filename if not provided
    if model_name is None:
        model_name = input_file.stem.replace('actor_', 'policy_')
    
    # Generate Sequential checkpoint
    print(f"Generating Sequential checkpoint...")
    output_content = generate_sequential_checkpoint(layers, model_name)
    
    # Write output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_content)
    
    print(f"Successfully converted checkpoint:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {model_name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert MLP checkpoint to Sequential format for firmware deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Convert a PPO checkpoint for firmware
    python convert_checkpoint_to_firmware.py controller/actor.h controller/actor_firmware.h
    
    # With custom model name
    python convert_checkpoint_to_firmware.py controller/actor.h controller/actor_firmware.h --name "PPO_return_878"
'''
    )
    parser.add_argument('input', help='Input MLP checkpoint file (.h)')
    parser.add_argument('output', help='Output Sequential checkpoint file (.h)')
    parser.add_argument('--name', '-n', help='Model name for metadata', default=None)
    
    args = parser.parse_args()
    
    success = convert_checkpoint(args.input, args.output, args.name)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
