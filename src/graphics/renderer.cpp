#include "renderer.h"

#include "graphics/webgpu_context.h"

#if defined(OPENXR_SUPPORT)

#include "xr/openxr/openxr_context.h"

#include "xr/dawnxr/dawnxr_internal.h"

#include "graphics/backend_include.h"

#elif defined(WEBXR_SUPPORT)

#include "xr/webxr/webxr_context.h"

#endif

#include "graphics/texture.h"
#include "framework/camera/camera.h"
#include "framework/nodes/light_3d.h"
#include "graphics/mesh.h"
#include "graphics/pipeline.h"
#include "graphics/shader.h"
#include "graphics/debug/renderdoc_capture.h"
#include "graphics/renderer_storage.h"

#include "shaders/mesh_forward.wgsl.gen.h"
#include "shaders/AABB_shader.wgsl.gen.h"
#include "shaders/mesh_shadow.wgsl.gen.h"


#include "framework/parsers/parse_scene.h"
#include "framework/nodes/mesh_instance_3d.h"
#include "framework/nodes/gs_node.h"
#include "framework/camera/camera_2d.h"
#include "framework/camera/flyover_camera.h"
#include "framework/camera/orbit_camera.h"
#include "framework/input.h"
#include "framework/ui/io.h"

#include <algorithm>

#include "shaders/quad_mirror.wgsl.gen.h"
#include "shaders/gaussian_splatting/gs_render.wgsl.gen.h"
#include "glm/gtx/quaternion.hpp"


#include "shaders/gbuffer_lighting_pass.wgsl.gen.h"
#include "shaders/gamma_pass.wgsl.gen.h"
#include "shaders/blur_compute.wgsl.gen.h"
#include "shaders/black_and_white.wgsl.gen.h"
#include "shaders/contrast_compute.wgsl.gen.h"

#include "spdlog/spdlog.h"

Renderer* Renderer::instance = nullptr;

Renderer::Renderer(const sRendererConfiguration& config)
{
    instance = this;

    renderer_storage = new RendererStorage();

#ifdef _DEBUG
    RenderdocCapture::init();
#endif

    webgpu_context = new WebGPUContext();

    Pipeline::webgpu_context = webgpu_context;
    Surface::webgpu_context = webgpu_context;
    Texture::webgpu_context = webgpu_context;

#if defined(OPENXR_SUPPORT)
    OpenXRContext* openxr_context = new OpenXRContext();
    is_xr_available = openxr_context->create_instance();

    xr_context = openxr_context;
#elif defined(WEBXR_SUPPORT)
    spdlog::info("Creating WebXR context");

    WebXRContext* webxr_context = new WebXRContext();
    is_xr_available = webxr_context->query_session_supported();

    xr_context = webxr_context;
#endif

    set_required_limits(config.required_limits);
    set_required_features(config.features);
}

Renderer::~Renderer()
{
    delete webgpu_context;

#ifdef XR_SUPPORT
    delete xr_context;
#endif
}

int Renderer::pre_initialize(GLFWwindow* window, bool use_mirror_screen)
{
    this->use_mirror_screen = use_mirror_screen;

    webgpu_context->window = window;
    webgpu_context->create_instance();

    Shader::set_custom_define("MAX_LIGHTS", MAX_LIGHTS);

    eye_depth_textures = new Texture[EYE_COUNT];
    multisample_textures = new Texture[EYE_COUNT];

#ifndef __EMSCRIPTEN__
    renderdoc_capture = new RenderdocCapture();
#endif

    return 0;
}

int Renderer::initialize()
{
    static WGPUFuture adapter_future = { 0 };
    static WGPUFuture device_future = { 0 };

    if (initialized) {
        return 0;
    }

    if (!webgpu_context->adapter) {
        if (adapter_future.id == 0) {
            adapter_future = webgpu_context->request_adapter(xr_context, is_xr_available);
        }
        webgpu_context->process_events();
        return 1;
    }

    if (webgpu_context->adapter && !webgpu_context->device) {
        if (device_future.id == 0) {
            // The engine needs FloatFilterable as a default
            required_features.push_back(WGPUFeatureName_Float32Filterable);
            device_future = webgpu_context->request_device(required_features);
        }
        webgpu_context->process_events();
        return 1;
    }

    bool create_screen_swapchain = true;

    if (is_xr_available) {
        create_screen_swapchain = use_mirror_screen;
    }

    // NOTE: breakpoint here for initial compute debugging in Metal
    if (webgpu_context->adapter && webgpu_context->device) {
        if (webgpu_context->initialize(create_screen_swapchain)) {
            spdlog::error("Could not initialize WebGPU context");
            return 1;
        }
    }

    spdlog::info("Renderer initialized");

    initialized = true;

    return 0;
}

int Renderer::post_initialize()
{
    webgpu_context->print_device_info();

#ifdef XR_SUPPORT

    xr_context->z_near = z_near;
    xr_context->z_far = z_far;

    if (is_xr_available && !xr_context->init(webgpu_context)) {
        spdlog::error("Could not initialize XR context");
        is_xr_available = false;
    }

    if (is_xr_available) {
        webgpu_context->render_width = xr_context->viewport.z;
        webgpu_context->render_height = xr_context->viewport.w;
    }
#endif

    if (!is_xr_available) {
        webgpu_context->render_width = webgpu_context->screen_width;
        webgpu_context->render_height = webgpu_context->screen_height;
    }

    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    global_command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    if (!irradiance_texture) {
        irradiance_texture = RendererStorage::get_texture("data/textures/environments/sky.hdr");
    }

    quad_surface.create_quad(2.0f, 2.0f);

    init_gbuffers();
	init_deferred_light_buffer();

    init_depth_buffers();

    init_lighting_bind_group();
    init_camera_bind_group();

    init_multisample_textures();

    init_timestamp_queries();

    init_deferred_lightpass();
	init_post_processing_textures();
	init_gamma_pass();
	init_compute_post_process(shaders::blur_compute::source, shaders::blur_compute::path, shaders::blur_compute::libraries, "box_blur");
	init_render_post_process(shaders::black_and_white::source, shaders::black_and_white::path, shaders::black_and_white::libraries);
	init_compute_post_process(shaders::contrast_compute::source, shaders::contrast_compute::path, shaders::contrast_compute::libraries, "contrast_compute");

    init_post_process_bindgroups();




#if defined(OPENXR_SUPPORT) && defined(USE_MIRROR_WINDOW)
    if (is_xr_available) {
        init_mirror_pipeline();
    }
#endif

    WGPUTextureFormat swapchain_format = is_xr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    RenderPipelineDescription desc = { .topology = WGPUPrimitiveTopology_TriangleStrip };

    WGPUBlendState* blend_state = new WGPUBlendState;
    blend_state->color = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_SrcAlpha,
            .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha,
    };
    blend_state->alpha = {
            .operation = WGPUBlendOperation_Add,
            .srcFactor = WGPUBlendFactor_Zero,
            .dstFactor = WGPUBlendFactor_One,
    };

    color_target.blend = blend_state;

    desc.depth_write = WGPUOptionalBool_False;
    desc.blending_enabled = true;
    desc.sample_count = msaa_count;

    // TODO( CONVERT GS TO DEFERRED
    gs_render_shader = RendererStorage::get_shader_from_source(shaders::gs_render::source, shaders::gs_render::path, shaders::gs_render::libraries);
    gs_render_pipeline.create_render_async(gs_render_shader, &color_target, 1u, desc);

    shadow_material = new Material();
    shadow_material->set_type(MATERIAL_UNLIT);
    //shadow_material->set_cull_type(CULL_FRONT);
    shadow_material->set_fragment_write(false);
    shadow_material->set_shader(RendererStorage::get_shader_from_source(shaders::mesh_shadow::source, shaders::mesh_shadow::path, shaders::mesh_shadow::libraries, shadow_material));

    // set initial memory size to avoid resizing every push_back
    current_render_list_size = 128;

    // Orthographic camera for ui rendering

    camera_2d = new Camera2D();
    camera_2d->set_orthographic(0.0f, static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height), 0.0f, -1.0f, 1.0f);

    //selected_mesh_aabb = parse_mesh("data/meshes/cube/aabb_cube.obj", false);

    //Material* AABB_material = new Material();
    //AABB_material->set_color(glm::vec4(0.8f, 0.3f, 0.9f, 1.0f));
    //AABB_material->set_transparency_type(ALPHA_BLEND);
    //AABB_material->set_cull_type(CULL_NONE);
    //AABB_material->set_type(MATERIAL_UNLIT);
    //AABB_material->set_shader(RendererStorage::get_shader_from_source(shaders::AABB_shader::source, shaders::AABB_shader::path, AABB_material));
    //selected_mesh_aabb->set_surface_material_override(selected_mesh_aabb->get_surface(0), AABB_material);

    return 0;
}

void Renderer::clean()
{

#if defined(XR_SUPPORT)

    xr_context->clean();

#if defined(USE_MIRROR_WINDOW)
    if (is_xr_available) {
        for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
            swapchain_uniforms[i].destroy();
            wgpuBindGroupRelease(swapchain_bind_groups[i]);
        }
    }

#endif // XR_SUPPORT
#endif // USE_MIRROR_WINDOW

    uint8_t num_textures = is_xr_available ? 2 : 1;
    for (int i = 0; i < num_textures; ++i)
    {
        wgpuTextureViewRelease(eye_depth_texture_view[i]);
    }

    RendererStorage::clean_registered_pipelines();

    wgpuBindGroupRelease(render_camera_bind_group);
    wgpuBindGroupRelease(render_camera_bind_group_2d);
    wgpuBindGroupRelease(shadow_camera_bind_group);
    wgpuBindGroupRelease(gbuffers_resolve_bindgroup);
	wgpuBindGroupRelease(gbuffers_light_pass_camera_bind_group);
	wgpuBindGroupRelease(post_processingAB);
	wgpuBindGroupRelease(post_processingBA);
	wgpuBindGroupRelease(post_processingAB_render);
	wgpuBindGroupRelease(post_processingBA_render); 
	wgpuBindGroupRelease(post_process_a_to_gamma_bindgroup);
	wgpuBindGroupRelease(post_process_b_to_gamma_bindgroup);

    camera_uniform.destroy();
    camera_2d_uniform.destroy();
    shadow_camera_uniform.destroy();

    for (int i = 0; i < RENDER_LIST_COUNT; ++i) {
        render_instances_data.instances_data_uniforms[i].destroy();

        if (render_instances_data.instances_bind_groups[i]) {
            wgpuBindGroupRelease(render_instances_data.instances_bind_groups[i]);
        }

        shadow_instances_data.instances_data_uniforms[i].destroy();

        if (shadow_instances_data.instances_bind_groups[i]) {
            wgpuBindGroupRelease(shadow_instances_data.instances_bind_groups[i]);
        }
    }

    webgpu_context->destroy();

    delete renderer_storage;
    delete[] eye_depth_textures;
    delete[] multisample_textures;

    delete shadow_material;

    delete[] gbuffer_data.textures;
    delete[] gbuffer_data.texture_views;
    delete gbuffer_data.depth_texture;
	delete light_buffer_data.texture;
	delete BufferA.texture;
	delete BufferB.texture;
    //delete selected_mesh_aabb;

    // TODO: WHAT HAPPENS WITH TEXTUREVIEW

    delete gbuffer_lighting_pass_shader;
	delete gamma_pass_shader;
	delete[] shaders_post_processing;

#ifndef __EMSCRIPTEN__
    delete renderdoc_capture;
#endif

    if (camera_3d) {
        delete camera_3d;
    }

    if (camera_2d) {
        delete camera_2d;
    }
}

void Renderer::update(float delta_time)
{
#if defined(XR_SUPPORT)
    if (is_xr_available) {
        xr_context->update();
    }
#endif

    if (debug_this_frame) {
        RenderdocCapture::start_capture_frame();
    }

    // Create the command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {};
    global_command_encoder = wgpuDeviceCreateCommandEncoder(webgpu_context->device, &encoder_desc);

    if (!is_xr_available) {
        const auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse && !io.WantCaptureKeyboard && !IO::any_focus()) {
            camera_3d->update(delta_time);
        }
    }
    else {
        camera_3d->update(delta_time);
    }
}

void Renderer::render()
{
    WGPUTextureView screen_surface_texture_view;
    WGPUSurfaceTexture screen_surface_texture;

    if (!eye_depth_texture_view[0]) {
        spdlog::error("Can not render if depth buffer is not initialized");
        clear_renderables();

#ifdef XR_SUPPORT
        // TODO: use callback for webxr first frame instead
        if (is_xr_available) {
            glm::ivec4 viewport = xr_context->viewport;

            if (viewport.z != webgpu_context->render_width || viewport.w != webgpu_context->render_height) {

                webgpu_context->render_width = viewport.z;
                webgpu_context->render_height = viewport.w;

                resize_window(viewport.z, viewport.w);

#if defined(USE_MIRROR_WINDOW)
                init_mirror_pipeline();
#endif
            }
        }
#endif

        ImGui::Render();

        return;
    }

    if (!is_xr_available || use_mirror_screen) {

        wgpuSurfaceGetCurrentTexture(webgpu_context->surface, &screen_surface_texture);
        if (screen_surface_texture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal) {
            spdlog::error("Error getting swapchain texture");
            return;
        }

        screen_surface_texture_view = webgpu_context->create_texture_view(screen_surface_texture.texture, WGPUTextureViewDimension_2D, webgpu_context->swapchain_format);
    }

    update_lights();

    //render_shadow_maps();

    camera_data.exposure = exposure;
    camera_data.ibl_intensity = ibl_intensity;
    camera_data.screen_size = { webgpu_context->screen_width, webgpu_context->screen_height };

    std::vector<std::vector<sRenderData>> render_lists(RENDER_LIST_COUNT);

    if (!is_xr_available) {
        camera_data.right_controller_position = camera_data.eye;

        prepare_cull_instancing(*camera_3d, render_lists, render_instances_data);

        camera_data.eye = camera_3d->get_eye();
        camera_data.view_projection = camera_3d->get_view_projection();
        camera_data.view = camera_3d->get_view();
        camera_data.projection = camera_3d->get_projection();
		camera_data.inv_view_projection = glm::inverse(camera_3d->get_view_projection());

        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), 0, &camera_data, sizeof(sCameraData));

        render_camera_in_gbuffers(render_lists, screen_surface_texture_view, eye_depth_texture_view[EYE_LEFT], render_instances_data, render_camera_bind_group, true, "deferred_render_pass");

        resolve_gbuffers(light_buffer_data.texture_view, eye_depth_textures[EYE_LEFT].get_texture(), eye_depth_texture_view[EYE_LEFT], render_instances_data, gbuffers_light_pass_camera_bind_group, true, "deferred_light_pass");

        //to do, copy light buffer to bufferA
		webgpu_context->copy_texture_to_texture(light_buffer_data.texture->get_texture(), BufferA.texture->get_texture(), 0, 0, light_buffer_data.texture->get_size(), { 0, 0, 0 }, { 0, 0, 0 }, global_command_encoder);

        post_processing_bool = true;
        for (int i = 0; i < num_declared_post_process_passes; i++) {
            if (render_post_process[i]) {
				std::vector<WGPUBindGroup> bind_groups = {};
				render_post_processing(post_process_pass_pipeline[i], bind_groups, shaders_post_processing[i]->get_path());
			}
        }
		/*
        for (Pipeline pipeline : post_process_pass_pipeline) {
			std::vector<WGPUBindGroup> bind_groups = {};
			render_post_processing(pipeline, bind_groups, "pass");
        }*/
        
        
		render_gamma_correction(screen_surface_texture_view, "gamma correction pass");
        //render_camera(render_lists, screen_surface_texture_view, eye_depth_texture_view[EYE_LEFT], render_instances_data, render_camera_bind_group, true, "forward_render");
    }
#ifdef XR_SUPPORT
    else {

        xr_context->init_frame();

        camera_data.right_controller_position = Input::get_controller_position(HAND_RIGHT);

        // prepare eye cameras
        Camera cameras[EYE_COUNT];
        for (uint32_t eye_idx = 0; eye_idx < EYE_COUNT; eye_idx++) {
            cameras[eye_idx].set_eye(xr_context->per_view_data[eye_idx].position);
            cameras[eye_idx].set_view(xr_context->per_view_data[eye_idx].view_matrix, false);
            cameras[eye_idx].set_projection(xr_context->per_view_data[eye_idx].projection_matrix, false);
            cameras[eye_idx].set_view_projection(xr_context->per_view_data[eye_idx].view_projection_matrix);
        }

        Camera vr_camera;
        vr_camera.set_eye((cameras[EYE_LEFT].get_eye() + cameras[EYE_RIGHT].get_eye()) * 0.5f);

        // Interpolate view
        {
            glm::mat4 left_view = xr_context->per_view_data[EYE_LEFT].view_matrix;
            glm::mat4 right_view = xr_context->per_view_data[EYE_RIGHT].view_matrix;
            glm::mat4 combined_view = left_view * 0.5f + right_view * 0.5f;
            vr_camera.set_view(combined_view, false);
        }

        // Set new FOV in projection
        {
            glm::mat4 left_proj = xr_context->per_view_data[EYE_LEFT].projection_matrix;
            float aspect = left_proj[1][1] / left_proj[0][0];
            if (!isnan(aspect)) {
                float eye_fov = 2.0f * atan(1.0f / left_proj[1][1]);
                float combined_eye_tan = tan(eye_fov / 2.0f) * 2.0f; // assuming same fov for both eyes
                float combined_fov = 2.0f * atan(combined_eye_tan);
                vr_camera.set_projection(glm::perspective(combined_fov, aspect, xr_context->z_near, xr_context->z_far));
            }
        }

        prepare_cull_instancing(vr_camera, render_lists, render_instances_data);

        for (uint32_t eye_idx = 0; eye_idx < EYE_COUNT; eye_idx++) {
            xr_context->acquire_swapchain(eye_idx);

            camera_data.eye = cameras[eye_idx].get_eye();
            camera_data.view_projection = cameras[eye_idx].get_view_projection();
            camera_data.view = cameras[eye_idx].get_view();
            camera_data.projection = cameras[eye_idx].get_projection();

            /*camera_data.inv_view = glm::inverse(camera_data.view);
            camera_data.inv_projection = glm::inverse(camera_data.projection);
            camera_data.inv_view_projection = glm::inverse(camera_data.view_projection);*/

            wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_uniform.data), eye_idx * camera_buffer_stride, &camera_data, sizeof(sCameraData));

            render_camera(render_lists, xr_context->get_swapchain_view(eye_idx), eye_depth_texture_view[eye_idx], render_instances_data, render_camera_bind_group, true, "forward_render_xr", eye_idx, eye_idx);

            xr_context->release_swapchain(eye_idx);
        }

#if defined(USE_MIRROR_WINDOW)
        if (use_mirror_screen) {
            render_mirror(screen_surface_texture_view, custom_mirror_fbo_bind_group ? custom_mirror_fbo_bind_group : swapchain_bind_groups[xr_context->get_swapchain_image_index(0)]);
        }
#endif
    }
#endif
    // TODO(Juan): Re enable 2d rendering after converting teh shaders to deferred
    // Render 2D
    if (!is_xr_available || use_mirror_screen) {

        camera_2d_data.eye = camera_2d->get_eye();
        camera_2d_data.view_projection = camera_2d->get_view_projection();

        camera_2d_data.exposure = exposure;
        camera_2d_data.ibl_intensity = ibl_intensity;

        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(camera_2d_uniform.data), 0, &camera_2d_data, sizeof(sCameraData));

        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};
        if (msaa_count > 1) {
            render_pass_color_attachment.view = multisample_textures_views[EYE_LEFT];
            render_pass_color_attachment.resolveTarget = screen_surface_texture_view;
        }
        else {
            render_pass_color_attachment.view = screen_surface_texture_view;
        }

        render_pass_color_attachment.loadOp = WGPULoadOp_Load;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        render_pass_color_attachment.clearValue = WGPUColor{ clear_color.r, clear_color.g, clear_color.b, clear_color.a };

        WGPURenderPassDescriptor render_pass_descr = {};
        render_pass_descr.colorAttachmentCount = 1;
        render_pass_descr.colorAttachments = &render_pass_color_attachment;
        render_pass_descr.depthStencilAttachment = nullptr;

        // Create & fill the render pass (encoder)
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

        if (custom_pre_2d_pass) {
            custom_pre_2d_pass(render_pass, render_camera_bind_group_2d, custom_pass_user_data, 0);
        }

        render_2D(render_pass, render_lists, render_instances_data, render_camera_bind_group_2d);

        if (custom_post_2d_pass) {
            custom_post_2d_pass(render_pass, render_camera_bind_group_2d, custom_pass_user_data, 0);
        }

        wgpuRenderPassEncoderEnd(render_pass);
        wgpuRenderPassEncoderRelease(render_pass);

        ImGui::Render();

// TODO: remove the ifdef, IMGui brings a viewport issue that can not be fixed by setting the viewport via webgpu
#ifndef BACKEND_METAL
    // render imgui
        {
            WGPURenderPassColorAttachment color_attachments = {};
            color_attachments.view = screen_surface_texture_view;
            color_attachments.loadOp = WGPULoadOp_Load;
            color_attachments.storeOp = WGPUStoreOp_Store;
            color_attachments.clearValue = { 0.0, 0.0, 0.0, 0.0 };
            color_attachments.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

            WGPURenderPassDescriptor render_pass_desc = {};
            render_pass_desc.colorAttachmentCount = 1;
            render_pass_desc.colorAttachments = &color_attachments;
            render_pass_desc.depthStencilAttachment = nullptr;

            WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_desc);

            //wgpuRenderPassEncoderSetViewport(render_pass, 0.0f,0.0f,1600.0f,900.0f,0.0f,1.0f);

            webgpu_context->push_debug_group(pass, { "ImGui", WGPU_STRLEN });

            ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);

            webgpu_context->pop_debug_group(pass);

            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
        }
#endif
    }

    submit_global_command_encoder();

    if (RenderdocCapture::is_capture_started() && debug_this_frame) {
        RenderdocCapture::end_capture_frame();
        debug_this_frame = false;
    }

    if (!is_xr_available) {
        wgpuTextureViewRelease(screen_surface_texture_view);
        wgpuTextureRelease(screen_surface_texture.texture);
    }
#ifdef XR_SUPPORT
    else {
        xr_context->end_frame();

#ifdef WEBXR_SUPPORT
        for (uint32_t eye_idx = 0; eye_idx < EYE_COUNT; eye_idx++) {
            wgpuTextureViewRelease(xr_context->get_swapchain_view(eye_idx));
        }
#endif // WEBXR_SUPPORT

    }
#endif // XR_SUPPORT

#ifndef __EMSCRIPTEN__
    if (!is_xr_available || use_mirror_screen) {
        wgpuSurfacePresent(webgpu_context->surface);
    }

    if (timestamps_requested) {
        get_timestamps();
        timestamps_requested = false;
    }
#endif

    clear_renderables();
}

void Renderer::render_camera_in_gbuffers(const std::vector<std::vector<sRenderData>>& render_lists, WGPUTextureView framebuffer_view, WGPUTextureView depth_view,
    const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents, const std::string& pass_name, uint32_t eye_idx, uint32_t camera_offset) {

    WGPURenderPassColorAttachment gbuffer_attachments[MAX_GBUFFER_COUNT];

    assert(webgpu_context->gbuffer_format.GBUFFER_COUNT <= MAX_GBUFFER_COUNT && "Verify that we are not using too many gbuffers");

    for (uint32_t i = 0u; i < webgpu_context->gbuffer_format.GBUFFER_COUNT; i++) {
        gbuffer_attachments[i] = {};
        gbuffer_attachments[i].view = gbuffer_data.texture_views[i];
        gbuffer_attachments[i].loadOp = WGPULoadOp_Clear;
        gbuffer_attachments[i].storeOp = WGPUStoreOp_Store;
        gbuffer_attachments[i].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        gbuffer_attachments[i].clearValue = WGPUColor{ clear_color.r, clear_color.g, clear_color.b, clear_color.a };
    }

    WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};
    render_pass_depth_attachment.view = gbuffer_data.depth_texture_view;
    render_pass_depth_attachment.depthClearValue = 0.0f;
    render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Clear;
    render_pass_depth_attachment.depthStoreOp = WGPUStoreOp_Store;
    render_pass_depth_attachment.depthReadOnly = false;
    render_pass_depth_attachment.stencilClearValue = 0; // Stencil config necesary, even if unused
    render_pass_depth_attachment.stencilLoadOp = WGPULoadOp_Undefined;
    render_pass_depth_attachment.stencilStoreOp = WGPUStoreOp_Undefined;
    render_pass_depth_attachment.stencilReadOnly = true;
  

    WGPURenderPassDescriptor render_pass_descr = {};
    render_pass_descr.colorAttachmentCount = webgpu_context->gbuffer_format.GBUFFER_COUNT;
    render_pass_descr.colorAttachments = gbuffer_attachments;
    render_pass_descr.depthStencilAttachment = &render_pass_depth_attachment;
    render_pass_descr.label = { pass_name.c_str(), pass_name.length() };

#ifndef __EMSCRIPTEN__
    std::vector<WGPUPassTimestampWrites> timestampWrites(1);
    timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_pre_render").c_str());
    timestampWrites[0].querySet = timestamp_query_set;
    timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_render").c_str());

    render_pass_descr.timestampWrites = timestampWrites.data();
#endif

    // Create & fill the render pass (encoder)
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { pass_name.c_str(), WGPU_STRLEN });
#endif

    // Opaque render calls to fill the G-buffer
    {
#ifndef NDEBUG
        webgpu_context->push_debug_group(render_pass, { "Opaque gbuffer-pass", WGPU_STRLEN });
#endif

        render_render_list(render_pass, render_lists[RENDER_LIST_OPAQUE], RENDER_LIST_OPAQUE, instance_data, camera_bind_group, camera_offset);

#ifndef NDEBUG
        webgpu_context->pop_debug_group(render_pass);
#endif
    }

    // Lighing pass

    // Post process...

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif

    wgpuRenderPassEncoderEnd(render_pass);

    wgpuRenderPassEncoderRelease(render_pass);
}

void Renderer::resolve_gbuffers(WGPUTextureView framebuffer_view, WGPUTexture depth_texture, WGPUTextureView depth_view, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents, const std::string& pass_name, uint32_t eye_idx, uint32_t camera_offset) {
    WGPURenderPassColorAttachment render_attachment = {};
    render_attachment.view = framebuffer_view;
    render_attachment.loadOp = WGPULoadOp_Clear;
    render_attachment.storeOp = WGPUStoreOp_Store;
    render_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    render_attachment.clearValue = WGPUColor{ clear_color.r, clear_color.g, clear_color.b, clear_color.a };

    WGPURenderPassDescriptor render_pass_descr = {};
    render_pass_descr.colorAttachmentCount = 1u;
    render_pass_descr.colorAttachments = &render_attachment;
    render_pass_descr.depthStencilAttachment = nullptr;
    render_pass_descr.label = { pass_name.c_str(), pass_name.length() };

#ifndef __EMSCRIPTEN__
    std::vector<WGPUPassTimestampWrites> timestampWrites(1);
    timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_pre_render").c_str());
    timestampWrites[0].querySet = timestamp_query_set;
    timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_render").c_str());

    render_pass_descr.timestampWrites = timestampWrites.data();
#endif

    // Create & fill the render pass (encoder)
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { pass_name.c_str(), WGPU_STRLEN });
#endif

    {
#ifndef NDEBUG
        webgpu_context->push_debug_group(render_pass, { "Lighting pass", WGPU_STRLEN });
#endif
		light_pass_deferred_pipeline.set(render_pass);

        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, gbuffers_resolve_bindgroup, 0, nullptr);
		uint32_t zero_offset = 0;
		wgpuRenderPassEncoderSetBindGroup(render_pass, 1, camera_bind_group, 1, &zero_offset);
		//wgpuRenderPassEncoderSetBindGroup(render_pass, 1, camera_bind_group, 1, &camera_buffer_stride);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 2, lighting_bind_group, 0, nullptr);


        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_surface.get_vertex_buffer(), 0, quad_surface.get_vertices_byte_size());
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, quad_surface.get_vertex_data_buffer(), 0, quad_surface.get_interleaved_data_byte_size());

        wgpuRenderPassEncoderDraw(render_pass, quad_surface.get_vertex_count(), 1, 0, 0);

#ifndef NDEBUG
        webgpu_context->pop_debug_group(render_pass);
#endif
    }

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif

    wgpuRenderPassEncoderEnd(render_pass);

    // COPY DEPTH FROM GBUFFER TO THE SWAPCHAIN'S DEPTH
	webgpu_context->copy_texture_to_texture(gbuffer_data.depth_texture->get_texture(), depth_texture, 0, 0, gbuffer_data.depth_texture->get_size(), { 0, 0, 0 }, { 0, 0, 0 }, global_command_encoder);

    wgpuRenderPassEncoderRelease(render_pass);
}


void Renderer::render_gamma_correction(WGPUTextureView framebuffer_view,
    const std::string& pass_name) {
	WGPURenderPassColorAttachment render_attachment = {};
	render_attachment.view = framebuffer_view;
	render_attachment.loadOp = WGPULoadOp_Clear;
	render_attachment.storeOp = WGPUStoreOp_Store;
	render_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
	render_attachment.clearValue = WGPUColor{ 0, 0, 0, clear_color.a };

	WGPURenderPassDescriptor render_pass_descr = {};
	render_pass_descr.colorAttachmentCount = 1u;
	render_pass_descr.colorAttachments = &render_attachment;
	render_pass_descr.depthStencilAttachment = nullptr;
	render_pass_descr.label = { pass_name.c_str(), pass_name.length() };

#ifndef __EMSCRIPTEN__
	std::vector<WGPUPassTimestampWrites> timestampWrites(1);
	timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_pre_render").c_str());
	timestampWrites[0].querySet = timestamp_query_set;
	timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_render").c_str());

	render_pass_descr.timestampWrites = timestampWrites.data();
#endif
	// Create & fill the render pass (encoder)
	WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

#ifndef NDEBUG
	webgpu_context->push_debug_group(render_pass, { pass_name.c_str(), WGPU_STRLEN });
#endif

	{
#ifndef NDEBUG
		webgpu_context->push_debug_group(render_pass, { "Gamma correction pass", WGPU_STRLEN });
#endif
		gamma_correction_pipeline.set(render_pass);

	

        WGPUBindGroup temporal = post_processing_bool ? post_process_a_to_gamma_bindgroup : post_process_b_to_gamma_bindgroup;

        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, temporal, 0, nullptr);

		// Set vertex buffer while encoding the render pass
		wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_surface.get_vertex_buffer(), 0, quad_surface.get_vertices_byte_size());
		wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, quad_surface.get_vertex_data_buffer(), 0, quad_surface.get_interleaved_data_byte_size());

		wgpuRenderPassEncoderDraw(render_pass, quad_surface.get_vertex_count(), 1, 0, 0);

#ifndef NDEBUG
		webgpu_context->pop_debug_group(render_pass);
#endif
	}

#ifndef NDEBUG
	webgpu_context->pop_debug_group(render_pass);
#endif

	wgpuRenderPassEncoderEnd(render_pass);
	wgpuRenderPassEncoderRelease(render_pass);
}


void Renderer::render_post_processing(Pipeline *pipeline, std::vector<WGPUBindGroup> extras, const std::string &pass_name) {
	if (pipeline->is_compute_pipeline()) {

        WGPUComputePassDescriptor compute_pass_descriptor = {};
       	WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(global_command_encoder, &compute_pass_descriptor);
    #ifndef NDEBUG
		    webgpu_context->push_debug_group(compute_pass, { pass_name.c_str(), WGPU_STRLEN });
    #endif
        {

            pipeline->set(compute_pass);

            WGPUBindGroup temporal = post_processing_bool ? post_processingAB: post_processingBA;

            wgpuComputePassEncoderSetBindGroup(compute_pass, 0, temporal, 0, nullptr);
			for (int i = 0; i < extras.size();i++) {
                //for now, dynamic offset is alays 0
				wgpuComputePassEncoderSetBindGroup(compute_pass, i+1, extras[i], 0, nullptr);
            }
			wgpuComputePassEncoderDispatchWorkgroups(compute_pass, webgpu_context->gbuffer_format.width / 8, webgpu_context->gbuffer_format.height / 8, 1);
        }
    #ifndef NDEBUG
		    webgpu_context->pop_debug_group(compute_pass);
    #endif
			wgpuComputePassEncoderEnd(compute_pass);
			wgpuComputePassEncoderRelease(compute_pass);
	}

    //Render pass pipeline
    else

    {
		    WGPUTextureView target_view = post_processing_bool ? BufferB.texture_view : BufferA.texture_view;

            WGPURenderPassColorAttachment render_attachment = {};
		    render_attachment.view = target_view;
			render_attachment.loadOp = WGPULoadOp_Clear;
			render_attachment.storeOp = WGPUStoreOp_Store;
			render_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
			render_attachment.clearValue = WGPUColor{ 0, 0, 0, clear_color.a };

			WGPURenderPassDescriptor render_pass_descr = {};
			render_pass_descr.colorAttachmentCount = 1u;
			render_pass_descr.colorAttachments = &render_attachment;
			render_pass_descr.depthStencilAttachment = nullptr;
			render_pass_descr.label = { pass_name.c_str(), pass_name.length() };
        
#ifndef __EMSCRIPTEN__
			std::vector<WGPUPassTimestampWrites> timestampWrites(1);
			timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_pre_render").c_str());
			timestampWrites[0].querySet = timestamp_query_set;
			timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_render").c_str());

			render_pass_descr.timestampWrites = timestampWrites.data();
#endif
			// Create & fill the render pass (encoder)
			WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

#ifndef NDEBUG
			webgpu_context->push_debug_group(render_pass, { pass_name.c_str(), WGPU_STRLEN });
#endif

			{
				pipeline->set(render_pass);

				WGPUBindGroup temporal = post_processing_bool ? post_processingAB_render : post_processingBA_render;

				wgpuRenderPassEncoderSetBindGroup(render_pass, 0, temporal, 0, nullptr);
				for (int i = 0; i < extras.size(); i++) {
					//for now, dynamic offset is alays 0
					wgpuRenderPassEncoderSetBindGroup(render_pass, i + 1, extras[i], 0, nullptr);
				}
				// Set vertex buffer while encoding the render pass
				wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_surface.get_vertex_buffer(), 0, quad_surface.get_vertices_byte_size());
				wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, quad_surface.get_vertex_data_buffer(), 0, quad_surface.get_interleaved_data_byte_size());

				wgpuRenderPassEncoderDraw(render_pass, quad_surface.get_vertex_count(), 1, 0, 0);

			}

#ifndef NDEBUG
			webgpu_context->pop_debug_group(render_pass);
#endif

			wgpuRenderPassEncoderEnd(render_pass);
			wgpuRenderPassEncoderRelease(render_pass);

    }
	post_processing_bool = !post_processing_bool;
}

void Renderer::render_camera(const std::vector<std::vector<sRenderData>>& render_lists, WGPUTextureView framebuffer_view, WGPUTextureView depth_view,
    const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, bool render_transparents, const std::string& pass_name, uint32_t eye_idx, uint32_t camera_offset)
{
    {
        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};

        if (framebuffer_view) {
            if (msaa_count > 1) {
                render_pass_color_attachment.view = multisample_textures_views[eye_idx];
                render_pass_color_attachment.resolveTarget = framebuffer_view;
            }
            else {
                render_pass_color_attachment.view = framebuffer_view;
            }

            render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
            render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
            render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            render_pass_color_attachment.clearValue = WGPUColor{ clear_color.r, clear_color.g, clear_color.b, clear_color.a };
        }

        // Prepate the depth attachment
        WGPURenderPassDepthStencilAttachment render_pass_depth_attachment = {};

        if (depth_view) {
            render_pass_depth_attachment.view = depth_view;
            render_pass_depth_attachment.depthClearValue = 0.0f;
            render_pass_depth_attachment.depthLoadOp = WGPULoadOp_Clear;
            render_pass_depth_attachment.depthStoreOp = WGPUStoreOp_Store;
            render_pass_depth_attachment.depthReadOnly = false;
            render_pass_depth_attachment.stencilClearValue = 0; // Stencil config necesary, even if unused
            render_pass_depth_attachment.stencilLoadOp = WGPULoadOp_Undefined;
            render_pass_depth_attachment.stencilStoreOp = WGPUStoreOp_Undefined;
            render_pass_depth_attachment.stencilReadOnly = true;
        }

        WGPURenderPassDescriptor render_pass_descr = {};
        render_pass_descr.colorAttachmentCount = framebuffer_view ? 1 : 0;
        render_pass_descr.colorAttachments = framebuffer_view ? &render_pass_color_attachment : nullptr;
        render_pass_descr.depthStencilAttachment = depth_view ? &render_pass_depth_attachment : nullptr;
        render_pass_descr.label = { pass_name.c_str(), pass_name.length() };

#ifndef __EMSCRIPTEN__
        std::vector<WGPUPassTimestampWrites> timestampWrites(1);
        timestampWrites[0].beginningOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_pre_render").c_str());
        timestampWrites[0].querySet = timestamp_query_set;
        timestampWrites[0].endOfPassWriteIndex = timestamp(global_command_encoder, (pass_name + "_render").c_str());

        render_pass_descr.timestampWrites = timestampWrites.data();
#endif
        // Create & fill the render pass (encoder)
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

#ifndef NDEBUG
        webgpu_context->push_debug_group(render_pass, { pass_name.c_str(), WGPU_STRLEN });
#endif

        if (custom_pre_opaque_pass) {
            custom_pre_opaque_pass(render_pass, camera_bind_group, custom_pass_user_data, camera_offset * camera_buffer_stride);
        }

        render_opaque(render_pass, render_lists, instance_data, camera_bind_group, camera_offset * camera_buffer_stride);

        if (custom_post_opaque_pass) {
            custom_post_opaque_pass(render_pass, camera_bind_group, custom_pass_user_data, camera_offset * camera_buffer_stride);
        }

        if (render_transparents) {
            if (custom_pre_transparent_pass) {
                custom_pre_transparent_pass(render_pass, camera_bind_group, custom_pass_user_data, camera_offset * camera_buffer_stride);
            }

            render_transparent(render_pass, render_lists, instance_data, camera_bind_group, camera_offset * camera_buffer_stride);

            if (custom_post_transparent_pass) {
                custom_post_transparent_pass(render_pass, camera_bind_group, custom_pass_user_data, camera_offset * camera_buffer_stride);
            }

            render_splats(render_pass, render_lists, instance_data, camera_bind_group, camera_offset * camera_buffer_stride);
        }

#ifndef NDEBUG
        webgpu_context->pop_debug_group(render_pass);
#endif

        wgpuRenderPassEncoderEnd(render_pass);

        wgpuRenderPassEncoderRelease(render_pass);
    }
}

bool Renderer::get_xr_available()
{
#ifdef XR_SUPPORT
    return is_xr_available;
#else
    return false;
#endif
}

bool Renderer::get_use_mirror_screen()
{
#if defined(USE_MIRROR_WINDOW)
    return use_mirror_screen;
#else
    return false;
#endif
}

void Renderer::process_events()
{
    webgpu_context->process_events();
}

void Renderer::submit_global_command_encoder()
{
    WGPUCommandBufferDescriptor cmd_buff_descriptor = {};
    cmd_buff_descriptor.nextInChain = NULL;
    cmd_buff_descriptor.label = { "Command buffer", WGPU_STRLEN };

    resolve_query_set(global_command_encoder, 0);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(global_command_encoder, &cmd_buff_descriptor);

    wgpuQueueSubmit(webgpu_context->device_queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(global_command_encoder);

}

void Renderer::set_camera_params(eCameraType camera_type, const glm::vec3& camera_eye, const glm::vec3& camera_center)
{
    this->camera_type = camera_type;

    Camera* old_camera = camera_3d;
    if (camera_type == CAMERA_FLYOVER) {
        camera_3d = new FlyoverCamera();
    }
    else if (camera_type == CAMERA_ORBIT) {
        camera_3d = new OrbitCamera();
    }

    camera_3d->set_perspective(glm::radians(45.0f), webgpu_context->screen_width / static_cast<float>(webgpu_context->screen_height), z_near, z_far);

    if (old_camera) {
        camera_3d->look_at(old_camera->get_eye(), old_camera->get_center(), old_camera->get_up());
    }
    else {
        camera_3d->look_at(camera_eye, camera_center, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    camera_3d->set_mouse_sensitivity(0.003f);
    camera_3d->set_speed(0.5f);

    delete old_camera;
}

eCameraType Renderer::get_camera_type()
{
    return camera_type;
}

void Renderer::set_custom_pass_user_data(void* user_data)
{
    this->custom_pass_user_data = user_data;
}


void Renderer::init_gbuffers() {
    webgpu_context->gbuffer_format.width = webgpu_context->render_width;
    webgpu_context->gbuffer_format.height = webgpu_context->render_height;

    gbuffer_data.textures = new Texture[webgpu_context->gbuffer_format.GBUFFER_COUNT];
    gbuffer_data.texture_views = new WGPUTextureView[webgpu_context->gbuffer_format.GBUFFER_COUNT];
    gbuffer_data.depth_texture = new Texture();

    for (uint32_t i = 0u; i < webgpu_context->gbuffer_format.GBUFFER_COUNT; i++) {
        gbuffer_data.textures[i].create(WGPUTextureDimension_2D,
            webgpu_context->gbuffer_format.GBUFFER_FORMAT,
            { webgpu_context->gbuffer_format.width, webgpu_context->gbuffer_format.height, 1u },
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding,
            1u,
            1u,
            nullptr);
        gbuffer_data.texture_views[i] = gbuffer_data.textures[i].get_view();
    }

    gbuffer_data.depth_texture->create(WGPUTextureDimension_2D,
        WGPUTextureFormat_Depth32Float,
        { webgpu_context->gbuffer_format.width, webgpu_context->gbuffer_format.height, 1u },
        WGPUTextureUsage_CopySrc | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding,
        1u,
        1u,
        nullptr);
    gbuffer_data.depth_texture_view = gbuffer_data.depth_texture->get_view();

}

void Renderer::init_deferred_light_buffer() {
	light_buffer_data.texture = new Texture();
	WGPUTextureUsage;
	light_buffer_data.texture->create(
		WGPUTextureDimension_2D,
		WGPUTextureFormat_RGBA32Float,
		{ webgpu_context->gbuffer_format.width, webgpu_context->gbuffer_format.height, 1u },
		WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc,
        1u, 1u, nullptr
    );

    light_buffer_data.texture_view = light_buffer_data.texture->get_view();
} 

void Renderer::init_gamma_pass() {
    //create pipeline
    {
	    WGPUColorTargetState color_target = {};
	    color_target.format = webgpu_context->swapchain_format;
	    color_target.blend = nullptr;
	    color_target.writeMask = WGPUColorWriteMask_All;

	    gamma_pass_shader = RendererStorage::get_shader_from_source(shaders::gamma_pass::source, shaders::gamma_pass::path, shaders::gamma_pass::libraries);
		gamma_correction_pipeline.create_render(gamma_pass_shader, color_target, { .use_depth = false, .allow_msaa = false });
	}
	//create bindgroup
    {
		Uniform texture_uniform;
		texture_uniform.data = BufferA.texture_view;
		texture_uniform.binding = 0;
		std::vector<Uniform *> uniforms;
		uniforms.push_back(&texture_uniform);

		post_process_a_to_gamma_bindgroup = webgpu_context->create_bind_group(uniforms, gamma_pass_shader, 0); 
    }
	{
		Uniform texture_uniform;
		texture_uniform.data = BufferB.texture_view;
		texture_uniform.binding = 0;
		std::vector<Uniform *> uniforms;
		uniforms.push_back(&texture_uniform);

		post_process_b_to_gamma_bindgroup = webgpu_context->create_bind_group(uniforms, gamma_pass_shader, 0);
	}

}


void Renderer::init_post_processing_textures() {
	BufferA.texture = new Texture();
	BufferA.texture->create(
			WGPUTextureDimension_2D,
			WGPUTextureFormat_RGBA32Float,
			{ webgpu_context->gbuffer_format.width, webgpu_context->gbuffer_format.height, 1u },
			WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding |WGPUTextureUsage_RenderAttachment,
			1u, 1u, nullptr);
	BufferA.texture_view = BufferA.texture->get_view();


    BufferB.texture = new Texture();
	BufferB.texture->create(
			WGPUTextureDimension_2D,
			WGPUTextureFormat_RGBA32Float,
			{ webgpu_context->gbuffer_format.width, webgpu_context->gbuffer_format.height, 1u },
			WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding | WGPUTextureUsage_RenderAttachment,
			1u, 1u, nullptr);
	BufferB.texture_view = BufferB.texture->get_view();

}


void Renderer::init_compute_post_process(const char *source, const std::string &name, const std::vector<std::string> &libraries, const std::string &entry_point) {
	{
		
		shaders_post_processing[num_declared_post_process_passes] =
            RendererStorage::get_shader_from_source(source, name, libraries);
		std::vector<WGPUConstantEntry> constants;
		post_process_pass_pipeline[num_declared_post_process_passes] = new Pipeline();
		post_process_pass_pipeline[num_declared_post_process_passes]->create_compute(shaders_post_processing[num_declared_post_process_passes], entry_point, constants);
		render_post_process[num_declared_post_process_passes] = true;
		num_declared_post_process_passes++;
    }

}

void Renderer::init_render_post_process(const char *source, const std::string &name, const std::vector<std::string> &libraries) {
	{
		WGPUColorTargetState color_target = {};
		//post_process format
		color_target.format = WGPUTextureFormat_RGBA32Float;
		color_target.blend = nullptr;
		color_target.writeMask = WGPUColorWriteMask_All;

        shaders_post_processing[num_declared_post_process_passes] =
		    RendererStorage::get_shader_from_source(source, name, libraries);
		post_process_pass_pipeline[num_declared_post_process_passes] = new Pipeline();
		post_process_pass_pipeline[num_declared_post_process_passes]->create_render(shaders_post_processing[num_declared_post_process_passes], color_target, { .use_depth = false, .allow_msaa = false });
		render_post_process[num_declared_post_process_passes] = true;
		num_declared_post_process_passes++;
    }
}

void Renderer::init_deferred_lightpass() {
    // Create lighting pass pipeline
    {
        WGPUColorTargetState color_target = {};
        //lighting_pass format
		color_target.format = WGPUTextureFormat_RGBA32Float;
        color_target.blend = nullptr;
        color_target.writeMask = WGPUColorWriteMask_All;
		std::vector<Uniform *> uniforms;

        gbuffer_lighting_pass_shader = RendererStorage::get_shader_from_source(shaders::gbuffer_lighting_pass::source, shaders::gbuffer_lighting_pass::path, shaders::gbuffer_lighting_pass::libraries);
		light_pass_deferred_pipeline.create_render(gbuffer_lighting_pass_shader, color_target, { .use_depth = false, .allow_msaa = false });
    }
    // Create bindgroup
    {
        std::vector<Uniform*> uniforms;
        Uniform uniform_list[MAX_GBUFFER_COUNT + 1];
        uint32_t uniform_count = 0u;
        // For GBuffer attachments
        for (;uniform_count < webgpu_context->gbuffer_format.GBUFFER_COUNT; uniform_count++) {
            uniform_list[uniform_count].binding = uniform_count;
            uniform_list[uniform_count].data = gbuffer_data.texture_views[uniform_count];

            uniforms.push_back(&uniform_list[uniform_count]);
        }
        // For gbuffer depth attachment
        uniform_list[uniform_count].binding = uniform_count;
        uniform_list[uniform_count].data = gbuffer_data.depth_texture_view; 


        uniforms.push_back(&uniform_list[uniform_count++]);

        gbuffer_sampler_uniform.data = webgpu_context->create_sampler();
        gbuffer_sampler_uniform.binding = uniform_count;

        uniforms.push_back(&gbuffer_sampler_uniform);

        gbuffers_resolve_bindgroup = webgpu_context->create_bind_group(uniforms, gbuffer_lighting_pass_shader, 0u);
    }
    //create gbuffer camera bindgroup
    {
		std::vector<Uniform *> cam_uniforms = { &camera_uniform };
		gbuffers_light_pass_camera_bind_group = webgpu_context->create_bind_group(cam_uniforms, gbuffer_lighting_pass_shader, 1u);
    }
}

void Renderer::init_lighting_bind_group()
{
    // delete if already created
    if (std::holds_alternative<WGPUTextureView>(irradiance_texture_uniform.data)) {
        wgpuTextureViewRelease(std::get<WGPUTextureView>(irradiance_texture_uniform.data));
        wgpuSamplerRelease(std::get<WGPUSampler>(ibl_sampler_uniform.data));
        wgpuBindGroupRelease(lighting_bind_group);
    }
    else {
        // only created once
        brdf_lut_uniform.data = webgpu_context->brdf_lut_texture->get_view();
        brdf_lut_uniform.binding = 1;
    }

    if (irradiance_texture) {

        irradiance_texture_uniform.data = irradiance_texture->get_view(WGPUTextureViewDimension_Cube, 0, 6, 0, 6);
        irradiance_texture_uniform.binding = 0;

        ibl_sampler_uniform.data = webgpu_context->create_sampler(
            WGPUAddressMode_ClampToEdge,
            WGPUAddressMode_ClampToEdge,
            WGPUAddressMode_ClampToEdge,
            WGPUFilterMode_Linear,
            WGPUFilterMode_Linear,
            WGPUMipmapFilterMode_Linear,
            static_cast<float>(irradiance_texture->get_mipmap_count())
        );

        ibl_sampler_uniform.binding = 2;
    }

    if (std::holds_alternative<WGPUBuffer>(lights_buffer.data)) {
        wgpuBufferDestroy(std::get<WGPUBuffer>(lights_buffer.data));
        lights_buffer.data = {};
    }

    lights_buffer.data = webgpu_context->create_buffer(sizeof(sLightUniformData) * MAX_LIGHTS, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, &lights_uniform_data[0], "lights_buffer");
    lights_buffer.binding = 3;
    lights_buffer.buffer_size = sizeof(sLightUniformData) * MAX_LIGHTS;

    num_lights_buffer.data = webgpu_context->create_buffer(sizeof(int), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, &num_lights, "num_lights_buffer");
    num_lights_buffer.binding = 4;
    num_lights_buffer.buffer_size = sizeof(int);

    // Shadow maps
    {
        shadow_array_texture = webgpu_context->create_texture(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_Depth32Float,
            { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, MAX_LIGHTS },
            WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst,
            1,
            1,
            "shadow_array_texture"
        );

        shadow_maps_array.data = webgpu_context->create_texture_view(
            shadow_array_texture,
            WGPUTextureViewDimension_2DArray,
            WGPUTextureFormat_Depth32Float,
            WGPUTextureAspect_DepthOnly,
            0,
            1,
            0,
            MAX_LIGHTS,
            "shadow_depth_texture_view"
        );
        shadow_maps_array.binding = 5;

        // Shadowmap sampler
        shadow_sampler.data = webgpu_context->create_sampler(
            WGPUAddressMode_ClampToEdge,
            WGPUAddressMode_ClampToEdge,
            WGPUAddressMode_ClampToEdge,
            WGPUFilterMode_Linear,
            WGPUFilterMode_Linear,
            WGPUMipmapFilterMode_Linear,
            1.0f,
            1u,
            WGPUCompareFunction_Greater // reverse Z
        );
        shadow_sampler.binding = 6;
    }

    std::vector<Uniform*> uniforms = { &irradiance_texture_uniform, &brdf_lut_uniform, &ibl_sampler_uniform, &lights_buffer, &num_lights_buffer, &shadow_maps_array, &shadow_sampler };
    lighting_bind_group = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries), 3);
}

void Renderer::init_depth_buffers()
{
    if (webgpu_context->render_width == 0 || webgpu_context->render_height == 0) {
        spdlog::error("Can not create depth buffer with size ({}, {})", webgpu_context->render_width, webgpu_context->render_height);
        return;
    }

    uint8_t num_textures = is_xr_available ? 2 : 1;
    for (int i = 0; i < num_textures; ++i)
    {
        eye_depth_textures[i].create(
            WGPUTextureDimension_2D,
            WGPUTextureFormat_Depth32Float,
            { webgpu_context->render_width, webgpu_context->render_height, 1 },
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst,
            1, msaa_count, nullptr);

        if (eye_depth_texture_view[i]) {
            wgpuTextureViewRelease(eye_depth_texture_view[i]);
        }

        // Generate Texture views of depth buffers
        eye_depth_texture_view[i] = eye_depth_textures[i].get_view();
    }

    spdlog::trace("Depth buffers initialized with size ({}, {})", webgpu_context->render_width, webgpu_context->render_height);
}

void Renderer::init_multisample_textures()
{
    if (webgpu_context->render_width == 0 || webgpu_context->render_height == 0) {
        spdlog::error("Can not multisample textures with size ({}, {})", webgpu_context->render_width, webgpu_context->render_height);
        return;
    }

    WGPUTextureFormat swapchain_format = is_xr_available ? webgpu_context->xr_swapchain_format : webgpu_context->swapchain_format;

    uint8_t num_textures = is_xr_available ? 2 : 1;
    for (int i = 0; i < num_textures; ++i) {
        multisample_textures[i].create(
            WGPUTextureDimension_2D,
            swapchain_format,
            { webgpu_context->render_width, webgpu_context->render_height, 1 },
            WGPUTextureUsage_RenderAttachment,
            1, msaa_count, nullptr);

        if (multisample_textures_views[i]) {
            wgpuTextureViewRelease(multisample_textures_views[i]);
        }

        multisample_textures_views[i] = multisample_textures[i].get_view();
    }
}

void Renderer::init_timestamp_queries()
{
#ifndef __EMSCRIPTEN__
    timestamp_query_set = webgpu_context->create_query_set(maximum_query_sets);
    timestamp_query_buffer = webgpu_context->create_buffer(sizeof(uint64_t) * maximum_query_sets, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, nullptr);
#endif
}

void Renderer::init_post_process_bindgroups() {

    	{ //fragment shader bindgroups
		Shader *post_process_shader = RendererStorage::get_shader_from_source(shaders::black_and_white::source, shaders::black_and_white::path, shaders::black_and_white::libraries);
		std::vector<Uniform *> uniforms;
		Uniform texture_uniformA;
		texture_uniformA.data = BufferA.texture_view;
		texture_uniformA.binding = 0;
		uniforms.push_back(&texture_uniformA);

		post_processingAB_render = webgpu_context->create_bind_group(uniforms, post_process_shader, 0);

		uniforms.clear();

		Uniform texture_uniformB;
		texture_uniformB.data = BufferB.texture_view;
		texture_uniformB.binding = 0;

		uniforms.push_back(&texture_uniformB);

		post_processingBA_render = webgpu_context->create_bind_group(uniforms, post_process_shader, 0);

	}

    { //compute shader bindgroups
		
		Shader* post_process_shader = RendererStorage::get_shader_from_source(shaders::blur_compute::source, shaders::blur_compute::path, shaders::blur_compute::libraries);
	
		std::vector<Uniform *> uniforms;
		Uniform texture_uniformA;
		texture_uniformA.data = BufferA.texture_view;
		texture_uniformA.binding = 0;

		Uniform texture_uniformB;
		texture_uniformB.data = BufferB.texture_view;
		texture_uniformB.binding = 1;
		uniforms.push_back(&texture_uniformA);
		uniforms.push_back(&texture_uniformB);

		post_processingAB = webgpu_context->create_bind_group(uniforms, post_process_shader, 0);

		uniforms.clear();
		Uniform second_texture_uniformA;
		second_texture_uniformA.data = texture_uniformA.data;
		second_texture_uniformA.binding = 1;

		Uniform second_texture_uniformB;
		second_texture_uniformB.data = texture_uniformB.data;
		second_texture_uniformB.binding = 0;

		uniforms.push_back(&second_texture_uniformB);
		uniforms.push_back(&second_texture_uniformA);

		post_processingBA = webgpu_context->create_bind_group(uniforms, post_process_shader, 0);

        }
}

void Renderer::resolve_query_set(WGPUCommandEncoder encoder, uint8_t first_query)
{
#ifndef __EMSCRIPTEN__
    wgpuCommandEncoderResolveQuerySet(encoder, timestamp_query_set, first_query, query_index, timestamp_query_buffer, 0);
#endif
}

void Renderer::get_timestamps()
{
#ifndef __EMSCRIPTEN__
    auto read_callback = [&](const void* output_buffer, void* user_data) {
        const uint64_t* timestamps_buffer = reinterpret_cast<const uint64_t*>(output_buffer);
        uint8_t* query_index_cpy = static_cast<uint8_t*>(user_data);

        std::vector<float> time_diffs;
        for (int i = 0; i < *query_index_cpy; i += 2) {
            uint64_t diff = timestamps_buffer[i + 1] - timestamps_buffer[i];
            float milliseconds = (float)diff * 1e-6f;
            time_diffs.push_back(milliseconds);
        }

        last_frame_timestamps = time_diffs;

        delete query_index_cpy;
        };

    // copy query_index, otherwise it'd have been already modified when reading
    uint8_t* query_index_cpy = new uint8_t();
    *query_index_cpy = query_index;
    webgpu_context->read_buffer_async(timestamp_query_buffer, sizeof(uint64_t) * maximum_query_sets, read_callback, query_index_cpy);
#endif

}

void Renderer::set_msaa_count(uint8_t msaa_count, bool is_initial_value)
{
    if (is_initial_value) {
        this->msaa_count = msaa_count;
        return;
    }

    bool recreate = msaa_count != this->msaa_count && multisample_textures[0].get_texture() != nullptr;

    this->msaa_count = msaa_count;

    if (recreate) {
        init_depth_buffers();
        init_multisample_textures();
    }

    RendererStorage::reload_all_render_pipelines();
}

void Renderer::set_frustum_camera_paused(bool value)
{
    frustum_camera_paused = value;
}

bool Renderer::get_frustum_camera_paused()
{
    return frustum_camera_paused;
}

uint8_t Renderer::get_msaa_count()
{
    return msaa_count;
}

void Renderer::prepare_cull_instancing(const Camera& camera, std::vector<std::vector<sRenderData>>& render_lists, sInstanceData& instances_data, bool is_shadow_pass)
{
    if (!frustum_camera_paused) {
        frustum_cull.set_view_projection(camera.get_view_projection());
    }

    // Get all surfaces from entity meshes
    for (auto render_list_data : render_entity_list)
    {
        Mesh* mesh = render_list_data.mesh;
        glm::mat4x4 global_matrix = render_list_data.global_matrix;

        const std::vector<Surface*>& surfaces = mesh->get_surfaces();

        for (Surface* surface : surfaces) {

            Material* material_override = mesh->get_surface_material_override(surface);
            Material* material = material_override ? material_override : surface->get_material();

            bool material_is_2d = material->get_is_2D();

            if (is_shadow_pass) {
                material = shadow_material;
            }

            if (!material || !material->get_shader()) {
                continue;
            }

            if (is_shadow_pass && (!mesh->get_receive_shadows() || material_is_2d || material->get_transparency_type() == ALPHA_BLEND)) {
                continue;
            }

            if (!material_is_2d && mesh->get_frustum_culling_enabled()) {

                const AABB& surface_aabb = surface->get_aabb();

                AABB aabb_transformed = surface_aabb.transform(global_matrix);

                if (!is_inside_frustum(aabb_transformed.center - aabb_transformed.half_size, aabb_transformed.center + aabb_transformed.half_size)) {
                    continue;
                }
            }

            RendererStorage::instance->register_material_bind_group(webgpu_context, mesh, material);

            RendererStorage::register_render_pipeline(material);

            eRenderListType list = RENDER_LIST_OPAQUE;

            if (material_is_2d) {
                list = material->get_transparency_type() == ALPHA_BLEND ? RENDER_LIST_2D_TRANSPARENT : RENDER_LIST_2D;
            }
            else if (material->get_transparency_type() == ALPHA_BLEND) {
                list = RENDER_LIST_TRANSPARENT;
            }

            render_lists[list].push_back({ surface, 1, global_matrix, mesh, material });
        }
    }

    for (int i = 0; i < RENDER_LIST_COUNT; ++i) {

        instances_data.instances_data[i].clear();
        instances_data.instances_data[i].resize(render_lists[i].size());

        /*if (i != RENDER_LIST_TRANSPARENT)*/ {
            // Sort opaques render_list
            std::sort(render_lists[i].begin(), render_lists[i].end(), [](auto& lhs, auto& rhs) {

                Material* lhs_mat = lhs.material;
                Material* rhs_mat = rhs.material;

                bool equal_priority = lhs_mat->get_priority() == rhs_mat->get_priority();
                bool equal_surface = lhs.surface == rhs.surface;

                if (lhs_mat->get_priority() > rhs_mat->get_priority()) return true;
                if (equal_priority && lhs.surface > rhs.surface) return true;
                if (equal_priority && equal_surface && lhs_mat > rhs_mat) return true;

                return false;
            });
        }
        //else {
        //    // Sort transparent render_list by distance to camera
        //    std::sort(render_list[i].begin(), render_list[i].end(), [&](auto& lhs, auto& rhs) {
        //        glm::vec3 lhs_pos = glm::vec3(lhs.global_matrix[3]);
        //        glm::vec3 rhs_pos = glm::vec3(rhs.global_matrix[3]);

        //        float lhs_dist = glm::distance2(lhs_pos, camera_position);
        //        float rhs_dist = glm::distance2(rhs_pos, camera_position);

        //        if (lhs_dist > rhs_dist) return true;

        //        return false;
        //    });
        //}

        // Check instances
        {
            const Surface* prev_surface = nullptr;
            Material* prev_material = nullptr;

            uint32_t repeats = 0;
            for (uint32_t j = 0; j < render_lists[i].size(); ++j) {

                const sRenderData& render_data = render_lists[i][j];

                Material* material = render_data.material;

                // Repeated MeshInstance3D, must be instanced
                if (prev_surface == render_data.surface && prev_material == material && !material->get_is_2D()) {
                    repeats++;
                }
                else {
                    if (repeats > 0) {
                        for (uint32_t k = 1; k <= repeats; k++) {
                            render_lists[i][j - k].repeat = k;
                        }
                    }
                    repeats = 1;
                }

                prev_surface = render_data.surface;
                prev_material = material;

                // Fill instance_data
                instances_data.instances_data[i][j] = { render_data.global_matrix };
            }

            if (repeats > 0) {
                for (uint32_t k = 1; k <= repeats; k++) {
                    render_lists[i][render_lists[i].size() - k].repeat = k;
                }
            }
        }

        // Fill instance buffers
        uint32_t instances = static_cast<uint32_t>(instances_data.instances_data[i].size());

        if (instances > (instances_data.instances_data_uniforms[i].buffer_size / sizeof(sUniformData))) {

            //std::vector<sUniformData> default_data = { instances, { glm::mat4x4(1.0f), glm::vec4(1.0f) } };

            if (std::holds_alternative<WGPUBuffer>(instances_data.instances_data_uniforms[i].data)) {
                wgpuBufferDestroy(std::get<WGPUBuffer>(instances_data.instances_data_uniforms[i].data));
            }

            instances_data.instances_data_uniforms[i].data = webgpu_context->create_buffer(sizeof(sUniformData) * instances, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage, instances_data.instances_data[i].data(), "instance_mesh_buffer");
            instances_data.instances_data_uniforms[i].binding = 0;
            instances_data.instances_data_uniforms[i].buffer_size = sizeof(sUniformData) * instances;

            // Recreate bind groups
            std::vector<Uniform*> uniforms = { &instances_data.instances_data_uniforms[i] };
            Shader* prev_shader = nullptr;
            for (uint32_t j = 0; j < render_lists[i].size(); ) {

                const sRenderData& render_data = render_lists[i][j];

                if (instances_data.instances_bind_groups[i]) {
                    wgpuBindGroupRelease(instances_data.instances_bind_groups[i]);
                }

                instances_data.instances_bind_groups[i] = webgpu_context->create_bind_group(uniforms, render_data.material->get_shader(), 0);

                j += render_data.repeat;
            }

        }
        else
            if (instances > 0) {
                webgpu_context->update_buffer(std::get<WGPUBuffer>(instances_data.instances_data_uniforms[i].data), 0, instances_data.instances_data[i].data(), sizeof(sUniformData) * instances);
            }
    }
}

void Renderer::render_shadow_maps()
{
    camera_data.exposure = exposure;
    camera_data.ibl_intensity = ibl_intensity;
    camera_data.screen_size = { webgpu_context->screen_width, webgpu_context->screen_height };

    std::vector<std::vector<sRenderData>> render_lists(RENDER_LIST_COUNT);

    for (uint32_t light_idx = 0; light_idx < lights_with_shadow.size(); ++light_idx) {

        Light3D* light = lights_with_shadow[light_idx];

        const Camera& light_camera = light->get_light_camera();

        prepare_cull_instancing(light->get_light_camera(), render_lists, shadow_instances_data, true);

        // Update main 3d camera

        camera_data.eye = light_camera.get_eye();
        camera_data.view = light_camera.get_view();
        camera_data.projection = light_camera.get_projection();
        camera_data.view_projection = light_camera.get_view_projection();

        wgpuQueueWriteBuffer(webgpu_context->device_queue, std::get<WGPUBuffer>(shadow_camera_uniform.data), light_idx * camera_buffer_stride, &camera_data, sizeof(sCameraData));

        if (!light->get_shadow_depth_texture()) {
            light->create_shadow_data();
        }

        render_camera(render_lists, nullptr, light->get_shadow_depth_texture_view(), shadow_instances_data, shadow_camera_bind_group, false, "shadow_map", 0, light_idx);
    }

    // copy shadow maps (temp solution)
    {
        for (uint32_t light_idx = 0; light_idx < lights_with_shadow.size(); ++light_idx) {
            Light3D* light = lights_with_shadow[light_idx];
            const WGPUExtent3D& size = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
            webgpu_context->copy_texture_to_texture(light->get_shadow_depth_texture(), shadow_array_texture, 0, 0, size, { 0, 0, 0 }, { 0, 0, light_idx }, get_global_command_encoder());
        }
    }
}

void Renderer::render_render_list(WGPURenderPassEncoder render_pass, const std::vector<sRenderData>& render_list, int list_index, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride)
{
    const Pipeline* prev_pipeline = nullptr;

    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, instance_data.instances_bind_groups[list_index], 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, camera_bind_group, 1, &camera_buffer_stride);

    const Material* prev_material = nullptr;

    for (int i = 0; i < render_list.size(); ) {

        const sRenderData& render_data = render_list[i];

        const Material* material = render_data.material;

        if (!material) {
            assert(0);
            continue;
        }

        const Pipeline* pipeline = material->get_shader()->get_pipeline();

        assert(pipeline);

        if (pipeline != prev_pipeline) {
            if (!pipeline->set(render_pass)) {
                i += render_data.repeat;
                continue;
            }
        }

        // Not initialized
        if (render_data.surface->get_vertex_count() == 0) {
            spdlog::error("Skipping not initialized mesh");
            continue;
        }

        // Set bind groups

        if (material != prev_material && (material->get_fragment_write() || (!material->get_fragment_write() && material->get_use_skinning())))
        {
            wgpuRenderPassEncoderSetBindGroup(render_pass, 2, renderer_storage->get_material_bind_group(material), 0, nullptr);

            if ((!prev_material && material->get_type() == MATERIAL_PBR) ||
                (prev_material && prev_material->get_type() != MATERIAL_PBR && material->get_type() == MATERIAL_PBR)) {
                wgpuRenderPassEncoderSetBindGroup(render_pass, 3, lighting_bind_group, 0, nullptr);
            }

            prev_material = material;
        }

        //#ifndef NDEBUG
        //        webgpu_context->push_debug_group(render_pass, render_data.surface->get_name().c_str());
        //#endif

        if (material->get_type() == MATERIAL_UI) {
            WGPUBindGroup ui_bind_group = renderer_storage->get_ui_widget_bind_group(render_data.mesh_ref);
            if (ui_bind_group) {
                wgpuRenderPassEncoderSetBindGroup(render_pass, 3, ui_bind_group, 0, nullptr);
            }
        }

        // Set vertex buffer while encoding the render pass
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, render_data.surface->get_vertex_buffer(), 0, render_data.surface->get_vertices_byte_size());

        if (material->get_fragment_write()) {
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, render_data.surface->get_vertex_data_buffer(), 0, render_data.surface->get_interleaved_data_byte_size());
        }

        WGPUBuffer index_buffer = render_data.surface->get_index_buffer();

        if (index_buffer) {
            wgpuRenderPassEncoderSetIndexBuffer(render_pass, index_buffer, WGPUIndexFormat_Uint32, 0, render_data.surface->get_indices_byte_size());

            wgpuRenderPassEncoderDrawIndexed(render_pass, render_data.surface->get_index_count(), render_data.repeat, 0, 0, i);
        }
        else {
            wgpuRenderPassEncoderDraw(render_pass, render_data.surface->get_vertex_count(), render_data.repeat, 0, i);
        }


        //#ifndef NDEBUG
        //        webgpu_context->pop_debug_group(render_pass);
        //#endif

        prev_pipeline = pipeline;

        i += render_data.repeat;
    }
}

void Renderer::render_opaque(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride)
{
#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { "Opaque", WGPU_STRLEN });
#endif

    render_render_list(render_pass, render_lists[RENDER_LIST_OPAQUE], RENDER_LIST_OPAQUE, instance_data, camera_bind_group, camera_buffer_stride);

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif
}

void Renderer::render_transparent(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride)
{
#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { "Transparent", WGPU_STRLEN });
#endif

    render_render_list(render_pass, render_lists[RENDER_LIST_TRANSPARENT], RENDER_LIST_TRANSPARENT, instance_data, camera_bind_group, camera_buffer_stride);

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif
}

void Renderer::render_splats(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group, uint32_t camera_buffer_stride)
{
#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { "Gaussian Splats", WGPU_STRLEN });
#endif

    for (GSNode* gs_node : gs_scenes_list) {

        if (!gs_render_pipeline.set(render_pass)) {
            continue;
        }

        wgpuRenderPassEncoderSetBindGroup(render_pass, 0, gs_node->get_render_bindgroup(), 0, nullptr);
        wgpuRenderPassEncoderSetBindGroup(render_pass, 1, camera_bind_group, 1, &camera_buffer_stride);

        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, gs_node->get_render_buffer(), 0, gs_node->get_splats_render_bytes_size());
        wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, gs_node->get_ids_buffer(), 0, gs_node->get_ids_render_bytes_size());

        wgpuRenderPassEncoderDraw(render_pass, 4, gs_node->get_splat_count(), 0, 0);
    }

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif
}

void Renderer::render_2D(WGPURenderPassEncoder render_pass, const std::vector<std::vector<sRenderData>>& render_lists, const sInstanceData& instance_data, WGPUBindGroup camera_bind_group)
{
#ifndef NDEBUG
    webgpu_context->push_debug_group(render_pass, { "2D", WGPU_STRLEN });
#endif

    render_render_list(render_pass, render_lists[RENDER_LIST_2D], RENDER_LIST_2D, instance_data, camera_bind_group);

    render_render_list(render_pass, render_lists[RENDER_LIST_2D_TRANSPARENT], RENDER_LIST_2D_TRANSPARENT, instance_data, camera_bind_group);

#ifndef NDEBUG
    webgpu_context->pop_debug_group(render_pass);
#endif
}

bool Renderer::is_inside_frustum(const glm::vec3& minp, const glm::vec3& maxp) const
{
    return frustum_cull.is_box_visible(minp, maxp);
}

uint8_t Renderer::timestamp(WGPUCommandEncoder encoder, const char* label)
{
    wgpuCommandEncoderWriteTimestamp(encoder, timestamp_query_set, query_index);
    queries_label_map[query_index] = std::string(label);

    assert(query_index + 1 < maximum_query_sets);

    return query_index++;
}

void Renderer::add_renderable(Mesh* mesh, const glm::mat4x4& global_matrix)
{
    if ((render_entity_list.size() + 1) >= current_render_list_size) {
        current_render_list_size <<= 1;
        render_entity_list.reserve(current_render_list_size);
    }

    render_entity_list.push_back({ mesh, global_matrix });
}

void Renderer::add_splat_scene(GSNode* gs_scene)
{
    gs_scenes_list.push_back(gs_scene);
}

void Renderer::clear_renderables()
{
    render_entity_list.clear();
    gs_scenes_list.clear();

    lights_with_shadow.clear();

    for (int i = 0; i < MAX_LIGHTS; ++i) {
        lights_uniform_data[i] = {};
    }

    num_lights = 0;
    query_index = 0;
}

void Renderer::update_lights()
{
    // update uniform data

    uint64_t buffer_size = sizeof(sLightUniformData) * num_lights;

    webgpu_context->update_buffer(std::get<WGPUBuffer>(lights_buffer.data), 0, &lights_uniform_data[0], buffer_size);

    webgpu_context->update_buffer(std::get<WGPUBuffer>(num_lights_buffer.data), 0, &num_lights, sizeof(int));
}

void Renderer::add_light(Light3D* new_light)
{
    sLightUniformData data;
    new_light->get_uniform_data(data);
    lights_uniform_data[num_lights] = data;
    num_lights++;

    const LightType light_type = new_light->get_type();

    if (!new_light->get_cast_shadows()) {
        return;
    }

    if (light_type == LIGHT_DIRECTIONAL) {
        lights_with_shadow.push_back(new_light);
    }
}

void Renderer::resize_window(int width, int height)
{
    webgpu_context->create_swapchain(width, height);

    if (!is_xr_available) {
        webgpu_context->screen_width = width;
        webgpu_context->screen_height = height;
        webgpu_context->render_width = width;
        webgpu_context->render_height = height;

        if (camera_3d) {
            camera_3d->set_perspective(glm::radians(45.0f), webgpu_context->render_width / static_cast<float>(webgpu_context->render_height), z_near, z_far);
        }

        if (camera_2d) {
            camera_2d->set_orthographic(0.0f, static_cast<float>(webgpu_context->render_width), static_cast<float>(webgpu_context->render_height), 0.0f, -1.0f, 1.0f);
        }
    }

    init_depth_buffers();
    init_multisample_textures();
}

void Renderer::set_irradiance_texture(Texture* texture)
{
    irradiance_texture = texture;

    init_lighting_bind_group();
}

std::string Renderer::get_shader_name_post_process(int index) {
	return shaders_post_processing[index]->get_path();
}

void Renderer::swap_post_process(int a, int b) {

	std::swap(shaders_post_processing[a], shaders_post_processing[b]);
	std::swap(render_post_process[a], render_post_process[b]);
	std::swap(post_process_pass_pipeline[a], post_process_pass_pipeline[b]);
}


#ifdef XR_SUPPORT
XRContext* Renderer::get_xr_context()
{
    return (is_xr_available ? xr_context : nullptr);
}
#endif

WebGPUContext* Renderer::get_webgpu_context()
{
    return webgpu_context;
}

GLFWwindow* Renderer::get_glfw_window()
{
    return webgpu_context->window;
};

#if defined(USE_MIRROR_WINDOW)

void Renderer::init_mirror_pipeline()
{
    mirror_shader = RendererStorage::get_shader_from_source(shaders::quad_mirror::source, shaders::quad_mirror::path, shaders::quad_mirror::libraries);

    WGPUTextureFormat swapchain_format = webgpu_context->swapchain_format;

    WGPUColorTargetState color_target = {};
    color_target.format = swapchain_format;
    color_target.blend = nullptr;
    color_target.writeMask = WGPUColorWriteMask_All;

    // Generate uniforms from the swapchain
    for (uint8_t i = 0; i < xr_context->get_num_images_per_swapchain(); i++) {
        Uniform swapchain_uni;

        swapchain_uni.data = xr_context->get_swapchain_view(EYE_LEFT, i);
        swapchain_uni.binding = 0;

        swapchain_uniforms.push_back(swapchain_uni);
    }

    linear_sampler_uniform.data = webgpu_context->create_sampler(WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge, WGPUFilterMode_Linear, WGPUFilterMode_Linear);
    linear_sampler_uniform.binding = 1;

    // Generate bindgroups from the swapchain
    for (uint8_t i = 0; i < swapchain_uniforms.size(); i++) {
        Uniform swapchain_uni;

        std::vector<Uniform*> uniforms = { &swapchain_uniforms[i], &linear_sampler_uniform };

        swapchain_bind_groups.push_back(webgpu_context->create_bind_group(uniforms, mirror_shader, 0));
    }

    mirror_pipeline.create_render(mirror_shader, color_target, { .use_depth = false, .allow_msaa = false });
}

void Renderer::render_mirror(WGPUTextureView screen_surface_texture_view, WGPUBindGroup displayed_fbo_bind_group)
{
    ImGui::Render();

    // Create & fill the render pass (encoder)
    {
        // Prepare the color attachment
        WGPURenderPassColorAttachment render_pass_color_attachment = {};
        render_pass_color_attachment.view = screen_surface_texture_view;
        render_pass_color_attachment.loadOp = WGPULoadOp_Clear;
        render_pass_color_attachment.storeOp = WGPUStoreOp_Store;
        render_pass_color_attachment.clearValue = WGPUColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
        render_pass_color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        WGPURenderPassDescriptor render_pass_descr = {};
        render_pass_descr.colorAttachmentCount = 1;
        render_pass_descr.colorAttachments = &render_pass_color_attachment;
        render_pass_descr.depthStencilAttachment = nullptr;

        {
            WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_descr);

            // Bind Pipeline
            if (!mirror_pipeline.set(render_pass)) {
                wgpuRenderPassEncoderEnd(render_pass);
                wgpuRenderPassEncoderRelease(render_pass);
                return;
            }

            // Set binding group
            wgpuRenderPassEncoderSetBindGroup(render_pass, 0, displayed_fbo_bind_group, 0, nullptr);

            // Set vertex buffer while encoding the render pass
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, quad_surface.get_vertex_buffer(), 0, quad_surface.get_vertices_byte_size());
            wgpuRenderPassEncoderSetVertexBuffer(render_pass, 1, quad_surface.get_vertex_data_buffer(), 0, quad_surface.get_interleaved_data_byte_size());

            // Submit drawcall
            wgpuRenderPassEncoderDraw(render_pass, 6, 1, 0, 0);

            wgpuRenderPassEncoderEnd(render_pass);
            wgpuRenderPassEncoderRelease(render_pass);
        }
    }

    // render imgui
    {
        WGPURenderPassColorAttachment color_attachments = {};
        color_attachments.view = screen_surface_texture_view;
        color_attachments.loadOp = WGPULoadOp_Load;
        color_attachments.storeOp = WGPUStoreOp_Store;
        color_attachments.clearValue = { 0.0, 0.0, 0.0, 0.0 };
        color_attachments.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;

        WGPURenderPassDescriptor render_pass_desc = {};
        render_pass_desc.colorAttachmentCount = 1;
        render_pass_desc.colorAttachments = &color_attachments;
        render_pass_desc.depthStencilAttachment = nullptr;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(global_command_encoder, &render_pass_desc);

        ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }
}

#endif

void Renderer::init_camera_bind_group()
{
    camera_buffer_stride = std::max(static_cast<uint32_t>(sizeof(sCameraData)), webgpu_context->required_limits.minUniformBufferOffsetAlignment);

    camera_uniform.data = webgpu_context->create_buffer(camera_buffer_stride * EYE_COUNT, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_buffer");
    camera_uniform.binding = 0;
    camera_uniform.buffer_size = sizeof(sCameraData);

    camera_2d_uniform.data = webgpu_context->create_buffer(sizeof(sCameraData), WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "camera_2d_buffer");
    camera_2d_uniform.binding = 0;
    camera_2d_uniform.buffer_size = sizeof(sCameraData);

    shadow_camera_uniform.data = webgpu_context->create_buffer(camera_buffer_stride * shadow_uniform_buffer_size, WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform, nullptr, "shadow_camera_buffer");
    shadow_camera_uniform.binding = 0;
    shadow_camera_uniform.buffer_size = sizeof(sCameraData);

    std::vector<Uniform*> uniforms = { &camera_uniform };
    render_camera_bind_group = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries), 1);

    uniforms = { &shadow_camera_uniform };
    shadow_camera_bind_group = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_shadow::source, shaders::mesh_shadow::path, shaders::mesh_shadow::libraries), 1);

    WGPUBindGroupLayoutEntry entry = {};
    entry.binding = 0;
    entry.buffer.type = WGPUBufferBindingType_Uniform;
    entry.buffer.hasDynamicOffset = true;
    entry.visibility = WGPUShaderStage_Compute;

    uniforms = { &camera_uniform };

    WGPUBindGroupLayout compute_camera_bind_group_layout = webgpu_context->create_bind_group_layout({ entry });
    compute_camera_bind_group = webgpu_context->create_bind_group(uniforms, compute_camera_bind_group_layout);

    uniforms = { &camera_2d_uniform };
    render_camera_bind_group_2d = webgpu_context->create_bind_group(uniforms, RendererStorage::get_shader_from_source(shaders::mesh_forward::source, shaders::mesh_forward::path, shaders::mesh_forward::libraries), 1);
}

glm::vec3 Renderer::get_camera_eye()
{
#if defined(XR_SUPPORT)
    if (is_xr_available) {
        return xr_context->per_view_data[0].position; // return left eye
    }
#endif

    return camera_3d->get_eye();
}

glm::vec3 Renderer::get_camera_front()
{
#if defined(XR_SUPPORT)
    if (is_xr_available) {
        glm::mat4x4 view = xr_context->per_view_data[0].view_matrix; // use left eye
        return { view[2].x, view[2].y, -view[2].z };
    }
#endif

    Camera* camera = get_camera();
    return glm::normalize(camera->get_center() - camera->get_eye());
}
