#include "sky.hpp"
#include "theme.hpp"
#include "../geometry.hpp"
#include "../../sim_constants.hpp"
#include <cmath>
#include <utility>

namespace renderer {

namespace {

constexpr RColor kSunDisc { 255, 240, 170, 255 };
constexpr RColor kMoon    { 215, 215, 215, 255 };

RVec3 add(const RVec3& a, const RVec3& b, float k) { return { a.x + b.x*k, a.y + b.y*k, a.z + b.z*k }; }

} // namespace

void Sky::Init() {
    Mesh m = geom::buildSphere(1.0f, 10, 20);
    sphere_ = wire::Upload(m.verts, m.idx, wire::FeatureEdges(m));
}

void Sky::Shutdown() { wire::Destroy(sphere_); }

void Sky::Draw(wire::Pipeline& p, const SkyFrame& f, const RCamera& cam) {
    lines_.clear();

    // A circle of radius r about c, facing the camera, as k segments.
    auto ring = [&](const RVec3& c, float r, int k, RColor col) {
        RVec3 d  = rmath::normalize(rmath::sub(c, cam.position));
        RVec3 rt = rmath::normalize(rmath::cross(d, std::fabs(d.y) < 0.99f ? RVec3{ 0, 1, 0 } : RVec3{ 1, 0, 0 }));
        RVec3 up = rmath::cross(rt, d);
        for (int i = 0; i < k; ++i) {
            float a0 = (float)consts::TAU * i / k, a1 = (float)consts::TAU * (i + 1) / k;
            lines_.push_back({ add(add(c, rt, r * cosf(a0)), up, r * sinf(a0)), col });
            lines_.push_back({ add(add(c, rt, r * cosf(a1)), up, r * sinf(a1)), col });
        }
        return std::pair{ rt, up };
    };

    // Sun: solid disc, then eight rays.
    p.Fill(sphere_, rmath::placeSphere(f.sun, f.sun_radius), kSunDisc);
    auto [rt, up] = ring(f.sun, f.sun_radius * 1.35f, 32, theme::kCaution);
    for (int i = 0; i < 8; ++i) {
        float a = (float)consts::TAU * (i + 0.5f) / 8;
        RVec3 dir = add({ rt.x * cosf(a), rt.y * cosf(a), rt.z * cosf(a) }, up, sinf(a));
        lines_.push_back({ add(f.sun, dir, f.sun_radius * 1.7f), theme::kCaution });
        lines_.push_back({ add(f.sun, dir, f.sun_radius * 2.6f), theme::kCaution });
    }

    // Moon: hidden-line globe lit by the sun, and its full outline.
    const RMat4 moon = rmath::placeSphere(f.moon, f.moon_radius);
    p.Fill(sphere_, moon);
    wire::Shading s;
    s.tint  = kMoon;
    s.light = f.sun_dir;
    s.night = 0.06f;
    p.Edges(sphere_, moon, s);
    ring(f.moon, f.moon_radius * 1.03f, 48, theme::kBorder);

    p.Lines(lines_.data(), lines_.size());
}

}
