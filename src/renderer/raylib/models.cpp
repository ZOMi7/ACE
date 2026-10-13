#include "models.hpp"
#include "theme.hpp"
#include "../geometry.hpp"
#include "../../sim_constants.hpp"
#include <cmath>

namespace renderer {

namespace {

constexpr int kSides = 16;   // hull/bell facets: 16 longerons read cleanly as wire

// cm_dist moves as propellant burns. The hull is built nose-at-origin and slid
// into place per frame, so it isn't part of the cache key.
bool sameDims(const RocketDims& a, const RocketDims& b) {
    return a.length == b.length && a.radius == b.radius && a.engine_dist == b.engine_dist &&
           a.nose_length == b.nose_length && a.sections == b.sections;
}

float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

RVec3 along(const RVec3& p, const RVec3& dir, double metres) {
    float k = (float)(metres * consts::M_TO_KM);
    return { p.x + dir.x * k, p.y + dir.y * k, p.z + dir.z * k };
}

// Rotation about local +Z (a cone's axis), for spinning the plume layers.
RMat4 spinZ(float a) {
    RMat4 m = rmath::identity();
    m.m[0] = cosf(a); m.m[1] = sinf(a);
    m.m[4] = -sinf(a); m.m[5] = cosf(a);
    return m;
}

RVec3 unit(const RVec3& v) { return rmath::normalize(v); }

// Additive, see-through, no depth write: plume and explosion layers.
Material glow(RColor c, float alpha) {
    Material m;
    m.color       = { c.r, c.g, c.b, (unsigned char)(clamp01(alpha) * 255.0f) };
    m.blend       = BlendMode::Additive;
    m.depth_write = false;
    return m;
}

} // namespace

void RocketModel::Ensure(RenderBackend& b, const RocketDims& dims) {
    if (cone_ == 0) {   // first use: the dimension-independent meshes
        cone_    = b.CreateMesh(geom::buildCone(12));
        diamond_ = b.CreateMesh(geom::buildSphere(1.0f, 2, 4));
        shell_   = b.CreateMesh(geom::buildSphere(1.0f, 8, 12));
    }
    if (hull_ != 0 && sameDims(dims_, dims)) { dims_ = dims; return; }
    for (const HullBell& e : cache_) {
        if (sameDims(e.dims, dims)) { hull_ = e.hull; bell_ = e.bell; dims_ = dims; return; }
    }
    HullBell e = buildHullBell(b, dims);
    cache_.push_back(e);
    hull_ = e.hull; bell_ = e.bell; dims_ = dims;
}

RocketModel::HullBell RocketModel::buildHullBell(RenderBackend& b, const RocketDims& d) const {
    const RColor kBody = theme::kPrimary;        // hull
    const RColor kNose = theme::kPrimary;        // nose
    const RColor kTrim = theme::kSelect;         // collar
    const RColor kBell = { 190, 190, 190, 255 }; // nozzle

    // Body frame: +Z = nose, tip at the origin. The nose is the sim's nosecone,
    // sitting on top of the stack (L doesn't include it), so it keeps its size
    // across staging, even when only a short payload stage is left under it.
    const float noseLen   = (float)d.nose_length;
    const float nose_base = -noseLen;

    // The stages left, each in its own shape, stacked down from the nose. A plain
    // cylinder of the widest radius stands in if the shapes aren't given.
    std::vector<HullSection> stack = d.sections;
    if (stack.empty()) stack.push_back({ HullSection::Cylinder, d.radius, d.length });

    Mesh hull;
    float top = nose_base;
    for (size_t i = stack.size(); i-- > 0;) {
        const float r = (float)stack[i].radius, len = (float)stack[i].length, bot = top - len;
        if (stack[i].kind == HullSection::Cone) {
            geom::appendFrustum(hull, bot, top, r, 0.0f, kSides, kBody, true, false);
        } else if (stack[i].kind == HullSection::Sphere) {
            std::vector<RVec3> arc;   // pole to pole
            for (int k = 0; k <= 8; ++k) {
                float a = (float)consts::TAU * k / 16;
                arc.push_back({ bot + len * 0.5f * (1.0f - cosf(a)), r * sinf(a), 0 });
            }
            geom::appendRevolve(hull, arc, kSides, kBody, false);
        } else {
            // Barrel in stacked sections, so the wire shows ring frames about every
            // three diameters, capped at both ends so the stage joints show.
            int rings = (int)lroundf(len / (r * 6.0f));
            rings = rings < 1 ? 1 : (rings > 8 ? 8 : rings);
            for (int k = 0; k < rings; ++k)
                geom::appendFrustum(hull, bot + len * k / rings, bot + len * (k + 1) / rings, r, r,
                                    kSides, kBody, k == 0, k == rings - 1);
        }
        top = bot;
    }

    // Thin collar where the nose meets the stack, then the nose: a cone with
    // straight sides to a sharp tip, as wide as the top stage.
    if (noseLen > 0.0f) {
        const float radius = (float)stack.back().radius;
        geom::appendFrustum(hull, nose_base - radius * 0.18f, nose_base, radius * 1.03f, radius,
                            kSides, kTrim, false, false);
        geom::appendFrustum(hull, nose_base, 0.0f, radius, 0.0f, kSides, kNose, false, false);
    }

    MeshHandle hullH = b.CreateMesh(hull);

    // Engine bell in the gimbal-pivot frame: throat at z=0, flaring to the exit
    // 1.5 m down (where the renderer puts the nozzle). Open at both ends.
    Mesh bell;
    std::vector<RVec3> bp;
    const int BS = 4;
    const float rT = (float)stack.front().radius * 0.4f, rE = (float)stack.front().radius * 1.15f;
    for (int i = BS; i >= 0; --i) {   // exit -> throat, so z increases
        float u = (float)i / BS;
        bp.push_back({ -1.5f * u, rT + (rE - rT) * powf(u, 1.5f), 0 });
    }
    geom::appendRevolve(bell, bp, kSides, kBell, /*capBase=*/false);
    MeshHandle bellH = b.CreateMesh(bell);

    return { d, hullH, bellH };
}

void RocketModel::Draw(RenderBackend& b, const RocketFrame& f) const {
    // Destroyed: the explosion replaces the intact hull + plume entirely.
    if (f.exploded) { drawDetonation(b, f); return; }

    // The hull mesh is built nose-at-origin; cm_dist is measured from the top of
    // the stack, below the nosecone, so slide it down by both to land the CoM at
    // st.r (which f.hull maps to the mesh origin).
    RMat4 hullM = rmath::mul(f.hull, rmath::translate({ 0, 0, (float)(f.dims.cm_dist + f.dims.nose_length) }));

    Material solid;
    solid.lit = true;   // lit = the backend applies aerodynamic heating
    b.DrawModel(hull_, hullM, solid);
    if (!f.has_engine) return;   // engineless stage (e.g. nosecone): no bell, thrust line or plume
    b.DrawModel(bell_, f.bell, solid);

    drawThrustAxis(b, f);
    if (f.firing) drawPlume(b, f);
}

void RocketModel::drawThrustAxis(RenderBackend& b, const RocketFrame& f) const {
    // The thrust line out of the gimbal pivot, beside a dashed reference along
    // the body axis: the angle between them is the nozzle deflection, readable
    // even at a degree or two where the bell itself barely moves.
    const RVec3  pivot = { f.bell.m[12], f.bell.m[13], f.bell.m[14] };
    const RVec3  aft   = unit({ -f.hull.m[8], -f.hull.m[9], -f.hull.m[10] });
    const double len   = dims_.length * 0.6 + 1.5;   // metres, past the bell exit

    std::vector<LineVertex> v;
    const int dashes = 10;
    const RColor ref = theme::withAlpha(theme::kInactive, 200);
    for (int i = 0; i < dashes; ++i) {
        v.push_back({ along(pivot, aft, len * i / dashes), ref });
        v.push_back({ along(pivot, aft, len * (i + 0.5) / dashes), ref });
    }
    const RColor thr = f.firing ? theme::kActive : theme::withAlpha(theme::kActive, 140);
    RVec3 tip = along(pivot, f.exhaust_dir, len);
    v.push_back({ pivot, thr });
    v.push_back({ tip, thr });
    // A short crossbar at the end marks it as the thrust line.
    RVec3 side = unit(rmath::cross(f.exhaust_dir, aft));
    if (rmath::length(side) < 0.5f) side = unit(rmath::cross(f.exhaust_dir, { 0, 1, 0 }));
    v.push_back({ along(tip, side, -dims_.radius * 0.6), thr });
    v.push_back({ along(tip, side,  dims_.radius * 0.6), thr });
    b.DrawLines(v.data(), v.size(), 1.0f);
}

void RocketModel::drawPlume(RenderBackend& b, const RocketFrame& f) const {
    // Nested wire cones streaming out of the nozzle along the exhaust direction.
    const double radius = dims_.radius;
    const double Lm     = dims_.length * 0.8 * f.thrust * f.flick;   // plume length, metres
    // Each layer turns slowly about the axis (at its own rate and direction), so
    // the wire flame looks alive rather than static.
    const float t = (float)b.Time();
    auto cone = [&](double r0_m, double len_m, RColor c, float alpha, float spin) {
        RVec3 apex = along(f.nozzle, f.exhaust_dir, len_m);
        RMat4 m = rmath::mul(rmath::orientCone(f.nozzle, apex, (float)(r0_m * consts::M_TO_KM)), spinZ(t * spin));
        b.DrawModel(cone_, m, glow(c, alpha * f.thrust));
    };
    cone(radius * 1.5,  Lm * 1.25, { 255, 110,  40, 255 }, 0.45f,  0.7f);   // outer haze
    cone(radius * 1.0,  Lm,        { 255, 160,  60, 255 }, 0.75f, -1.1f);   // flame body
    cone(radius * 0.55, Lm * 0.6,  { 255, 235, 180, 255 }, 0.90f,  1.9f);   // white-hot core

    // Mach diamonds: shock nodes that only form in atmosphere (over/under-
    // expanded nozzle), so they fade with altitude and down the plume.
    if (f.air <= 0.12f) return;
    const int   nD      = 5;
    const float spacing = (float)(radius * 1.7) * (0.15f + 0.5f * (1.0f - f.air));   // metres
    for (int k = 1; k <= nD; ++k) {
        float dist = spacing * k;
        if (dist > Lm * 0.95) break;
        float fade = 1.0f - (float)(k - 1) / nD;
        float sz   = (float)(radius * 0.45 * consts::M_TO_KM) * (0.6f + 0.4f * fade);
        b.DrawModel(diamond_, rmath::placeSphere(along(f.nozzle, f.exhaust_dir, dist), sz),
                    glow({ 215, 230, 255, 255 }, f.air * f.thrust * fade));
    }
}

void RocketModel::drawDetonation(RenderBackend& b, const RocketFrame& f) const {
    const float a      = f.det_time;    // seconds since detonation
    const float Dfire  = 1.5f;          // fireball lifetime
    const float Dtotal = 4.0f;          // incl. lingering smoke
    if (a > Dtotal) return;             // fully dissipated: the rocket is gone

    // Concentric wire shells about the rocket's position, a few tens of metres
    // across (floored so small rockets still read), same timeline as bgfx.
    const float Rmax = (float)(fmax(dims_.radius * 25.0, 30.0) * consts::M_TO_KM);   // km
    auto shell = [&](float radiusKm, RColor c, float alpha) {
        if (alpha <= 0.002f || radiusKm <= 0.0f) return;
        b.DrawModel(shell_, rmath::placeSphere(f.center, radiusKm), glow(c, alpha));
    };

    // Flash: a brief bright burst at t=0.
    if (a < 0.12f) {
        float p = a / 0.12f;
        shell(Rmax * (0.5f + 0.6f * p), { 255, 250, 235, 255 }, 1.0f - p);
    }

    // Fireball: nested layers, ease-out expansion, fading over Dfire.
    if (a < Dfire) {
        float p    = a / Dfire;
        float R    = Rmax * (1.0f - (1.0f - p) * (1.0f - p));   // fast then slowing
        float env  = 1.0f - p;
        float turb = 0.9f + 0.15f * f.flick;
        shell(R * 1.15f * turb, { 200,  70,  20, 255 }, env * 0.55f);   // smoky outer
        shell(R * 0.85f * turb, { 255, 130,  40, 255 }, env * 0.85f);   // flame body
        shell(R * 0.50f,        { 255, 230, 180, 255 }, 1.0f - p * p);  // hot core
    }

    // Shockwave: a fast thin front, gone by 0.6 s.
    if (a < 0.6f) {
        float p = a / 0.6f;
        shell(Rmax * (0.4f + 1.8f * p), { 200, 220, 255, 255 }, (1.0f - p) * 0.5f);
    }

    // Lingering smoke: a dim shell that slowly grows and fades out by Dtotal.
    const float s0 = Dfire * 0.6f;
    if (a > s0) {
        float q = clamp01((a - s0) / (Dtotal - s0));
        shell(Rmax * (1.0f + 0.6f * q), { 120, 80, 60, 255 }, (1.0f - q) * 0.45f);
    }
}

}
