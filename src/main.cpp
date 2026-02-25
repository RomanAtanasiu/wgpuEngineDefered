#include "engine/engine.h"
#include "graphics/renderer.h"

extern void get_engine_config(sEngineConfiguration& out_config);

int main()
{
    sEngineConfiguration configuration;

    get_engine_config(configuration);

    Engine* engine;
    engine = new Engine();
    

    Renderer* renderer;

    renderer = new Renderer();
    

    if (engine->initialize(renderer, configuration)) {
        return 1;
    }

    engine->start_loop();

    engine->clean();

    delete renderer;
    delete engine;

    return 0;
}
