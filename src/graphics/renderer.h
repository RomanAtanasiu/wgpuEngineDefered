#pragma once

#include "includes.h"
#include "graphics/uniforms_structs.h"
#include "graphics/uniform.h"
#include "framework/math/frustum_cull.h"
#include "graphics/surface.h"
#include "graphics/pipeline.h"

#include "framework/camera/camera.h"

#include "glm/mat4x4.hpp"

#include "backends/imgui_impl_wgpu.h"

#include <map>
#include <string>

#define MAX_LIGHTS 32u
#define SHADOW_MAP_SIZE 1024
#define MAX_POST_PROCESS_PASS 10u


class Camera;
class Texture;
class Surface;
class Light3D;
class RenderdocCapture;
class RendererStorage;
class Mesh;
class MeshInstance3D;
class GSNode;
struct GLFWwindow;
struct WebGPUContext;
struct XRContext;
struct sLightUniformData;

struct sRendererConfiguration {
    WGPULimits required_limits = {};
    std::vector<WGPUFeatureName> features;

    sRendererConfiguration() {
        required_limits.maxVertexAttributes = 4;
        required_limits.maxVertexBuffers = 1;
        required_limits.maxBindGroups = 2;
        required_limits.maxUniformBuffersPerShaderStage = 1;
        required_limits.maxUniformBufferBindingSize = 65536;
        required_limits.minUniformBufferOffsetAlignment = 256;
        required_limits.minStorageBufferOffsetAlignment = 256;
        required_limits.maxComputeInvocationsPerWorkgroup = 256;
        required_limits.maxSamplersPerShaderStage = 1;
        required_limits.maxDynamicUniformBuffersPerPipelineLayout = 1;

#if !defined(__EMSCRIPTEN__)
        features.push_back(WGPUFeatureName_TimestampQuery);
#endif
    }
};

class Renderer {

protected:

    XRContext*  xr_context;
    WebGPUContext*  webgpu_context;

    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_pre_opaque_pass = nullptr;
    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_post_opaque_pass = nullptr;
    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_pre_transparent_pass = nullptr;
    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_post_transparent_pass = nullptr;
    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_pre_2d_pass = nullptr;
    std::function<void(WGPURenderPassEncoder, WGPUBindGroup, void*, uint32_t)> custom_post_2d_pass = nullptr;

    void* custom_pass_user_data = nullptr;

    WGPUCommandEncoder global_command_encoder;

    Camera* camera_3d = nullptr;
    Camera* camera_2d = nullptr;

    // Render meshes with material color
    WGPUBindGroup render_camera_bind_group = nullptr;
    WGPUBindGroup shadow_camera_bind_group = nullptr;
    WGPUBindGroup compute_camera_bind_group = nullptr;
    WGPUBindGroup render_camera_bind_group_2d = nullptr;
	WGPUBindGroup gbuffers_light_pass_camera_bind_group = nullptr;



    Uniform camera_uniform;
    Uniform camera_2d_uniform;
    Uniform shadow_camera_uniform;

    uint32_t camera_buffer_stride = 0;

    Texture* irradiance_texture = nullptr;

    Texture*        eye_depth_textures = nullptr;
    WGPUTextureView eye_depth_texture_view[EYE_COUNT] = {};

    uint8_t msaa_count = 1;
    Texture* multisample_textures;
    WGPUTextureView multisample_textures_views[EYE_COUNT] = {};

    RendererStorage* renderer_storage;

    struct sGbuffers {
        Texture *textures = nullptr;
        WGPUTextureView *texture_views = nullptr;
        Texture* depth_texture = nullptr;
        WGPUTextureView depth_texture_view = nullptr;
    } gbuffer_data;

    struct sLightBuffer {
		Texture* texture = nullptr;
		WGPUTextureView texture_view = nullptr;
	} light_buffer_data;

    struct sBufferPostProcess{
		Texture *texture = nullptr;
		WGPUTextureView texture_view = nullptr;
    };
	sBufferPostProcess BufferA;
	sBufferPostProcess BufferB;
	bool post_processing_bool = true;

    WGPUBindGroup post_processingAB = nullptr;
	WGPUBindGroup post_processingBA = nullptr;

    WGPUBindGroup post_processingAB_render = nullptr;
	WGPUBindGroup post_processingBA_render = nullptr;

    WGPUBindGroup gbuffers_resolve_bindgroup = nullptr;
	WGPUBindGroup post_process_a_to_gamma_bindgroup = nullptr;
	WGPUBindGroup post_process_b_to_gamma_bindgroup = nullptr;

    struct sPostProcessData {
		Shader *shader = nullptr;
		Pipeline* pipeline = nullptr;
		bool activated = false;
		//int index;
		int id;
        std::vector<WGPUBindGroup> bindgroups;
		void add_bind_group(WGPUBindGroup bind_group) {bindgroups.push_back(bind_group);}
	};
	//std::list<int> post_processing_index;
	//std::array<int, MAX_POST_PROCESS_PASS> post_process_index;
	int post_process_index[MAX_POST_PROCESS_PASS];
    sPostProcessData post_process_data[MAX_POST_PROCESS_PASS];
	int smallest_id = 0;
	void update_smallest_id();
	int num_declared_post_process_passes = 0;

	WGPUBuffer blur_level_buffer = nullptr;
	int32_t blur_level = 5;

	Shader *gbuffer_lighting_pass_shader = nullptr;
	Shader* gamma_pass_shader = nullptr;

    Pipeline light_pass_deferred_pipeline;
	Pipeline gamma_correction_pipeline;


    Uniform  gbuffer_sampler_uniform;

    void render_post_processing(Pipeline *pipeline, std::vector<WGPUBindGroup> extras, const std::string &pass_name = "");

#ifndef __EMSCRIPTEN__
    RenderdocCapture* renderdoc_capture;
#endif

    Frustum frustum_cull;
    MeshInstance3D* selected_mesh_aabb = nullptr;

    bool is_xr_available        = false;
    bool use_mirror_screen      = false;
    bool use_custom_mirror      = false;

    glm::vec4 clear_color = { 0.1f, 0.1f, 0.1f, 1.0f };

    // inverted for reverse-z
    float z_near = 1000.0f;
    float z_far = 0.01f;

    struct sUniformData {
        glm::mat4x4 model;
    };

    struct sRenderListData {
        Mesh* mesh;
        glm::mat4x4 global_matrix;
    };

    struct sRenderData {
        Surface* surface;
        uint32_t repeat;
        glm::mat4x4 global_matrix;
        Mesh* mesh_ref;
        Material* material;
    };

    enum eRenderListType {
        RENDER_LIST_OPAQUE,
        RENDER_LIST_TRANSPARENT,
        RENDER_LIST_SPLATS,
        RENDER_LIST_2D,
        RENDER_LIST_2D_TRANSPARENT,
        RENDER_LIST_COUNT
    };

    struct sInstanceData {
        std::vector<sUniformData> instances_data[RENDER_LIST_COUNT];
        Uniform	instances_data_uniforms[RENDER_LIST_COUNT];
        WGPUBindGroup instances_bind_groups[RENDER_LIST_COUNT] = {};
    };



    struct sCameraData {
        glm::mat4x4 view_projection;
        glm::mat4x4 view;
        glm::mat4x4 projection;
        //glm::mat4x4 inv_view;
        //glm::mat4x4 inv_projection;
        glm::mat4x4 inv_view_projection;

        glm::vec3 eye = {};
        float exposure = 1.0f;

        glm::vec3 right_controller_position = {};
        float ibl_intensity = 1.0f;

        glm::vec2 screen_size;
		//glm::vec3 dummy;
		uint32_t show_depthbuffer = 0;
		uint32_t show_gbuffers = 0; 
		//uint32_t gbuffer_num;

        glm::vec4 dummy;
		glm::vec4 dummy_for_inv[12];
    };

    eCameraType camera_type = CAMERA_FLYOVER;
    sCameraData camera_data;
    sCameraData camera_2d_data;

    void render_shadow_maps();

    void render_render_list(WGPURenderPassEncoder render_pass, const std::vector<sRenderData>& render_list, int list_index, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);

    void init_camera_bind_group();

    void get_timestamps();

    sInstanceData render_instances_data;
    sInstanceData shadow_instances_data;

    std::vector<sUIData> instance_ui_data;
    Uniform	instance_ui_data_uniform;

    // Entities to be rendered this frame
    std::vector<sRenderListData> render_entity_list;
    uint32_t current_render_list_size = 32;

    // Gaussian Splatting scenes to render
    std::vector<GSNode*> gs_scenes_list;

    // Bind group for lighting

    WGPUBindGroup lighting_bind_group;

    // Indirect lighting

    Uniform irradiance_texture_uniform;
    Uniform brdf_lut_uniform;
    Uniform ibl_sampler_uniform;

    // Direct lighting

    sLightUniformData lights_uniform_data[MAX_LIGHTS];
    int num_lights = 0;

    Uniform lights_buffer;
    Uniform num_lights_buffer;
    Uniform shadow_maps_array;
    Uniform shadow_sampler;
    WGPUTexture shadow_array_texture;

    // Shadows

    uint32_t shadow_uniform_buffer_size = MAX_LIGHTS;
    std::vector<Light3D*> lights_with_shadow;

    Material* shadow_material;

    Pipeline gs_render_pipeline;
    Shader* gs_render_shader = nullptr;

    WGPUQuerySet timestamp_query_set;
    uint8_t maximum_query_sets = 16;

    WGPUBuffer timestamp_query_buffer;
    uint8_t query_index = 0;
    std::map<uint8_t, std::string> queries_label_map;
    std::vector<float> last_frame_timestamps;
    bool timestamps_requested = false;

    bool frustum_camera_paused = false;
    bool debug_this_frame = false;

    float exposure = 1.0f;
    float ibl_intensity = 1.0f;

    bool initialized = false;

    std::vector<WGPUFeatureName> required_features = { };

    uint32_t frame_counter = 0;

public:

    // Singleton
    static Renderer* instance;

    Renderer(const sRendererConfiguration& config = {});
    virtual ~Renderer();

    virtual int pre_initialize(GLFWwindow* window, bool use_mirror_screen = false);
    virtual int initialize();
    virtual int post_initialize();
    virtual void clean();

    bool is_initialized() { return initialized; }
    
    virtual void update(float delta_time);
    virtual void render();

    void render_camera(const std::vector<std::vector<sRenderData>>& render_lists, WGPUTextureView framebuffer_view, WGPUTextureView depth_view,
        const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents = true, const std::string& pass_name = "", uint32_t eye_idx = 0, uint32_t camera_offset = 0);

    void process_events();

    void submit_global_command_encoder();

    void set_camera_params(eCameraType camera_type, const glm::vec3& camera_eye, const glm::vec3& camera_center);
    eCameraType get_camera_type();

    void set_custom_pass_user_data(void* user_data);

    void increase_frame_counter() { frame_counter++; }
    uint32_t get_frame_counter() { return frame_counter; }

    void init_lighting_bind_group();
    WGPUBindGroup get_lighting_bind_group() { return lighting_bind_group; }
    WGPUBindGroup get_render_camera_bind_group() { return render_camera_bind_group; }
    WGPUBindGroup get_compute_camera_bind_group() { return compute_camera_bind_group; }

    void init_gbuffers();
	void init_deferred_light_buffer();
	void init_gamma_pass();

	void init_compute_post_process(const char *source, const std::string &name,
			const std::vector<std::string> &libraries, const std::string &entry_point, int* id = nullptr);
	void init_render_post_process(const char *source, const std::string &name,
			const std::vector<std::string> &libraries, int* id = nullptr);
    void init_deferred_lightpass();
	void init_post_processing_textures();
    void init_depth_buffers();
    void init_multisample_textures();
    void init_timestamp_queries();
	void init_post_process_bindgroups();

    void set_frustum_camera_paused(bool value);
    bool get_frustum_camera_paused();

    bool get_use_custom_mirror() { return use_custom_mirror; }

    void request_timestamps() { timestamps_requested = true; }

    WGPUCommandEncoder get_global_command_encoder() { return global_command_encoder; }

    void resolve_query_set(WGPUCommandEncoder encoder, uint8_t first_query);
    std::vector<float>& get_last_frame_timestamps() { return last_frame_timestamps; }

    void set_msaa_count(uint8_t msaa_count, bool is_initial_value = false);
    uint8_t get_msaa_count();

    bool is_inside_frustum(const glm::vec3& minp, const glm::vec3& maxp) const;

    void prepare_cull_instancing(const Camera& camera, std::vector<std::vector<sRenderData>>& render_lists, sInstanceData& instances_data, bool is_shadow_pass = false);
    void render_opaque(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);
    void render_transparent(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);
    void render_splats(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride = 0);
    void render_2D(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group);

    void render_camera_in_gbuffers(const std::vector<std::vector<sRenderData>>& render_lists, WGPUTextureView framebuffer_view, WGPUTextureView depth_view,
        const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents, const std::string& pass_name = "", uint32_t eye_idx = 0, uint32_t camera_offset = 0);

    void resolve_gbuffers(WGPUTextureView framebuffer_view, WGPUTexture depth_texture, WGPUTextureView depth_view,
        const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents, const std::string& pass_name = "", uint32_t eye_idx = 0, uint32_t camera_offset = 0);

    void render_gamma_correction(WGPUTextureView framebuffer_view,
        const std::string &pass_name = "");

    bool get_xr_available();
    bool get_use_mirror_screen();

    inline void set_exposure(float new_exposure) { exposure = new_exposure; }
    inline void set_ibl_intensity(float new_intensity) { ibl_intensity = new_intensity; }

    inline void toogle_frame_debug() { debug_this_frame = true; }

    float get_exposure() { return exposure; }
    float get_ibl_intensity() { return ibl_intensity; }

    inline Uniform* get_current_camera_uniform() { return &camera_uniform; }
    glm::vec3 get_camera_eye();
    glm::vec3 get_camera_front();

    // For the XR mirror screen
#if defined(USE_MIRROR_WINDOW)
    void render_mirror(WGPUTextureView screen_surface_texture_view, WGPUBindGroup displayed_fbo_bind_group);
    void init_mirror_pipeline();

    void set_custom_mirror_fbo_bind_group(WGPUBindGroup fbo_bind_group) { custom_mirror_fbo_bind_group = fbo_bind_group; }

    Pipeline mirror_pipeline;
    Shader* mirror_shader = nullptr;
    WGPUBindGroup custom_mirror_fbo_bind_group = nullptr;

    Uniform linear_sampler_uniform;

    Surface quad_surface;

    std::vector<Uniform> swapchain_uniforms;
    std::vector<WGPUBindGroup> swapchain_bind_groups;
#endif // USE_MIRROR_WINDOW

    uint8_t timestamp(WGPUCommandEncoder encoder, const char* label = "");

    WGPUQuerySet get_query_set() { return timestamp_query_set; }
    std::map<uint8_t, std::string>& get_queries_label_map() { return queries_label_map; }

    glm::vec4 get_clear_color() { return clear_color; }

#ifdef XR_SUPPORT
    XRContext* get_xr_context();
#endif
    WebGPUContext* get_webgpu_context();

    void set_required_features(std::vector<WGPUFeatureName> new_required_features) { required_features = new_required_features; };
    void set_required_limits(const WGPULimits& required_limits) { webgpu_context->required_limits = required_limits; }

    void add_renderable(Mesh* mesh_instance, const glm::mat4x4& global_matrix);
    void add_splat_scene(GSNode* gs_scene);
    void clear_renderables();

    void update_lights();
    void add_light(Light3D* new_light);

    virtual void resize_window(int width, int height);

    GLFWwindow* get_glfw_window();

    void set_irradiance_texture(Texture* texture);
    Texture* get_irradiance_texture() { return irradiance_texture; }

    Camera* get_camera() { return camera_3d; }

    void set_show_gbuffers(int a) { camera_data.show_gbuffers = a; }
	int get_show_gbuffers() { return camera_data.show_gbuffers; }

	void set_show_depthbuffer(int a) { camera_data.show_depthbuffer = a; }
	int get_show_depthbuffer() { return camera_data.show_depthbuffer; }

    int get_num_of_gbuffers() { return webgpu_context->gbuffer_format.GBUFFER_COUNT; }


    int get_num_of_post_process_passes() { return num_declared_post_process_passes; }

    int get_post_process_index_from_id(int id);
	int get_post_process_id_from_index(int index) const { return post_process_index[index]; }

    bool get_post_process_enabled_from_index(int index) { return post_process_data[index].activated; }
	bool get_post_process_enabled_from_id(int id) { return post_process_data[get_post_process_index_from_id(id)].activated; }

    void set_post_process_enabled_from_index(int index, bool value) {  post_process_data[index].activated = value; }
	void set_post_process_enabled_from_id(int id, bool value) { post_process_data[get_post_process_index_from_id(id)].activated = value; }

    std::string get_shader_name_post_process_from_index(int index);
	std::string get_shader_name_post_process_from_id(int id);

	void swap_post_process(int a, int b);
	int* get_blur_level() { return &blur_level; }
    void set_blur_level(int level) {
		blur_level = level;
		webgpu_context->update_buffer(blur_level_buffer, 0, &blur_level, sizeof(int32_t));
    }
};
