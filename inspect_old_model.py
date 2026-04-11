import argparse
import os
from collections import Counter
from typing import List, Optional

import numpy as np


def _shape_to_str(shape) -> str:
    if shape is None:
        return "?"
    try:
        if isinstance(shape, (list, tuple)):
            return "(" + ", ".join("None" if d is None else str(d) for d in shape) + ")"
        return str(shape)
    except Exception:
        return str(shape)


def _safe_count_params(weights) -> int:
    total = 0
    for w in weights:
        try:
            total += int(np.prod(w.shape))
        except Exception:
            pass
    return total


def _get_layer_output_shape(layer) -> str:
    try:
        return _shape_to_str(layer.output_shape)
    except Exception:
        pass

    try:
        out = layer.output
        if isinstance(out, (list, tuple)):
            shapes = []
            for x in out:
                s = getattr(x, "shape", None)
                shapes.append(_shape_to_str(tuple(s)) if s is not None else "?")
            return "[" + ", ".join(shapes) + "]"
        s = getattr(out, "shape", None)
        if s is not None:
            return _shape_to_str(tuple(s))
    except Exception:
        pass

    return "?"


def inspect_keras_model(model_path: str, fallback_builder: bool) -> List[str]:
    import tensorflow as tf

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("MODEL DETAIL REPORT (KERAS)")
    lines.append("=" * 80)
    lines.append(f"File: {model_path}")
    lines.append(f"Size (MB): {os.path.getsize(model_path) / (1024 * 1024):.3f}")

    model = None
    load_errors = []

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        lines.append("Load mode: full model")
    except Exception as e:
        load_errors.append(f"load_model failed: {e}")

    if model is None and fallback_builder:
        try:
            from CNN_Classification_Model.model import cnn_model

            model = cnn_model()
            model.load_weights(model_path)
            lines.append("Load mode: weights-only via CNN_Classification_Model.model.cnn_model")
        except Exception as e:
            load_errors.append(f"fallback cnn_model + load_weights failed: {e}")

    if model is None:
        lines.append("ERROR: Could not load model.")
        if load_errors:
            lines.append("Details:")
            for err in load_errors:
                lines.append(f"- {err}")
        return lines

    lines.append("")
    lines.append("Model summary:")
    summary_lines: List[str] = []
    model.summary(print_fn=summary_lines.append, expand_nested=True)
    lines.extend(summary_lines)

    total_params = _safe_count_params(model.weights)
    trainable_params = _safe_count_params(model.trainable_weights)
    non_trainable_params = _safe_count_params(model.non_trainable_weights)

    lines.append("")
    lines.append("Global stats:")
    lines.append(f"- Input shape : {_shape_to_str(getattr(model, 'input_shape', None))}")
    lines.append(f"- Output shape: {_shape_to_str(getattr(model, 'output_shape', None))}")
    lines.append(f"- Total params      : {total_params:,}")
    lines.append(f"- Trainable params  : {trainable_params:,}")
    lines.append(f"- Non-trainable     : {non_trainable_params:,}")

    lines.append("")
    lines.append("Per-layer details:")

    for idx, layer in enumerate(model.layers):
        layer_name = str(layer.name)[:28]
        layer_type = layer.__class__.__name__[:24]
        out_shape = _get_layer_output_shape(layer)
        params = int(layer.count_params())
        trainable = "yes" if getattr(layer, "trainable", False) else "no"
        lines.append(
            f"- [{idx:02d}] name={layer_name}; type={layer_type}; output_shape={out_shape}; params={params:,}; trainable={trainable}"
        )

    return lines


def _onnx_dims_to_str(tensor_type) -> str:
    dims = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return "(" + ", ".join(dims) + ")"


def inspect_onnx_model(model_path: str) -> List[str]:
    import onnx

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("MODEL DETAIL REPORT (ONNX)")
    lines.append("=" * 80)
    lines.append(f"File: {model_path}")
    lines.append(f"Size (MB): {os.path.getsize(model_path) / (1024 * 1024):.3f}")

    model = onnx.load(model_path)
    graph = model.graph

    total_params = 0
    for init in graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total_params += n

    lines.append("")
    lines.append("Global stats:")
    lines.append(f"- IR version   : {model.ir_version}")
    lines.append(f"- Opset imports: {', '.join(str(op.version) for op in model.opset_import)}")
    lines.append(f"- Nodes        : {len(graph.node)}")
    lines.append(f"- Initializers : {len(graph.initializer)}")
    lines.append(f"- Total params : {total_params:,}")

    lines.append("")
    lines.append("Inputs:")
    for x in graph.input:
        t = x.type.tensor_type
        elem_type = t.elem_type
        lines.append(f"- {x.name}: dtype={elem_type}, shape={_onnx_dims_to_str(t)}")

    lines.append("")
    lines.append("Outputs:")
    for y in graph.output:
        t = y.type.tensor_type
        elem_type = t.elem_type
        lines.append(f"- {y.name}: dtype={elem_type}, shape={_onnx_dims_to_str(t)}")

    op_counter = Counter(node.op_type for node in graph.node)
    lines.append("")
    lines.append("Operator histogram:")
    for op_type, count in sorted(op_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {op_type}: {count}")

    lines.append("")
    lines.append("Node list:")
    lines.append(f"{'#':>3} | {'Name':<35} | {'OpType':<20} | Inputs -> Outputs")
    lines.append("-" * 110)
    for idx, node in enumerate(graph.node):
        node_name = (node.name or "(no_name)")[:35]
        ins = ", ".join(node.input)
        outs = ", ".join(node.output)
        lines.append(f"{idx:>3} | {node_name:<35} | {node.op_type:<20} | {ins} -> {outs}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect detailed model information for Keras or ONNX files")
    parser.add_argument(
        "--model",
        default="h5_file/new_model_v2.h5",
        help="Path to model file (.h5, .keras, .onnx). Default: h5_file/new_model_v2.h5",
    )
    parser.add_argument(
        "--save",
        default="model_detail_report.txt",
        help="Path to save report text. Use empty string to disable file output.",
    )
    parser.add_argument(
        "--no-fallback-builder",
        action="store_true",
        help="Disable fallback loading via CNN_Classification_Model.model.cnn_model for .h5 weights-only files.",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return

    ext = os.path.splitext(model_path)[1].lower()
    if ext in {".h5", ".keras"}:
        lines = inspect_keras_model(model_path, fallback_builder=not args.no_fallback_builder)
    elif ext == ".onnx":
        lines = inspect_onnx_model(model_path)
    else:
        print(f"ERROR: Unsupported extension: {ext}")
        print("Supported: .h5, .keras, .onnx")
        return

    report = "\n".join(lines)
    print(report)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(report)
        print("\nSaved report to:", args.save)


if __name__ == "__main__":
    main()
