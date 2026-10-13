#pragma once

#include "wire.hpp"
#include "../scene.hpp"
#include <vector>

// The sun and moon at their real positions and sizes. The sun is a solid disc
// with a ring of rays, like a symbol on a chart; the moon is a wire globe lit
// from the sun, so it shows its phase, inside a faint outline of its full disc.
// The Earth hides either one when it passes behind.

namespace renderer {

class Sky {
public:
    void Init();       // needs the GL context
    void Shutdown();

    void Draw(wire::Pipeline& p, const SkyFrame& f, const RCamera& cam);

private:
    wire::GpuMesh           sphere_;   // unit globe, for both bodies
    std::vector<LineVertex> lines_;    // reused every frame
};

}
