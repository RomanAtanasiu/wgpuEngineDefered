#include "hash.h"

bool RenderPipelineKey::operator==(const RenderPipelineKey& other) const
{
    bool is_equal = other.color_target_count == color_target_count;
    if (!is_equal) {
        return false;
    }
    uint32_t min_count = std::min(other.color_target_count, color_target_count);
    for (uint32_t i = 0u; i < min_count; i++) {
        is_equal &= other.color_targets[i].format == color_targets[i].format;
        is_equal &= other.color_targets[i].writeMask == color_targets[i].writeMask;
        is_equal &= other.color_targets[i].blend == color_targets[i].blend;
    }
    return (shader == other.shader && is_equal
        && description.cull_mode == other.description.cull_mode
        && description.topology == other.description.topology
        && description.depth_read == other.description.depth_read
        && description.depth_write == other.description.depth_write
        && description.blending_enabled == other.description.blending_enabled
        && description.depth_compare == other.description.depth_compare
        && pipeline_layout == other.pipeline_layout);
}
