// Continuous-angle CPU reference for a fixed finite cloud; no transport cache.
// clang++ -O3 -std=c++20 -pthread cloud-reference.cpp -o reference.exe
// reference.exe <output-directory> [maximum-spp=4096] [replicates=8] [threads=4]
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

constexpr double pi = 3.14159265358979323846;
constexpr double extinction = 0.18547168941993848 * 0.15;
constexpr double albedo = 0.18547160197076002 / 0.18547168941993848;
constexpr double sunRadius = 0.266 * pi / 180;
constexpr int width = 32, height = 40;

struct Vec {
    double x, y, z;
    Vec operator+(Vec b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec operator-(Vec b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec operator/(Vec b) const { return {x/b.x, y/b.y, z/b.z}; }
};
double dot(Vec a, Vec b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec cross(Vec a, Vec b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
Vec normalize(Vec a) { return a * (1/std::sqrt(dot(a, a))); }
Vec orient(Vec axis, double mu, double phi) {
    Vec tangent = normalize(cross(std::abs(axis.z) < 0.9 ? Vec{0,0,1} : Vec{0,1,0}, axis));
    Vec bitangent = cross(axis, tangent);
    double radial = std::sqrt(std::max(0.0, (1-mu)*(1+mu)));
    return axis*mu + (tangent*std::cos(phi) + bitangent*std::sin(phi))*radial;
}

// SplitMix64, one independent stream per replicate and fixed pixel.
struct Random {
    uint64_t state;
    double uniform() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        return (double(z >> 12) + 0.5) * 0x1p-52;
    }
};

struct Phase {
    double gh, gd, alpha, wd;
    static Phase droplets(double diameter) {
        return {std::exp(-0.0990567/(diameter-1.67154)),
                std::exp(-2.20679/(diameter+3.91029)-0.428934),
                std::exp(3.62489-8.29288/(diameter+5.52825)),
                std::exp(-0.599085/(diameter-0.641583)-0.665888)};
    }
    double hg(double g, double mu) const {
        double den = (1-g)*(1-g) + 2*g*(1-mu);
        return (1-g*g)/(4*pi*den*std::sqrt(den));
    }
    double value(double mu) const {
        return (1-wd)*hg(gh, mu) + wd*hg(gd, mu)*(1+alpha*mu*mu)/(1+alpha*(1+2*gd*gd)/3);
    }
    double hgSample(double g, Random& rng) const {
        if (g == 0) return 2*rng.uniform()-1;
        double v = (1-g*g)/(1-g+2*g*rng.uniform());
        return std::clamp((1+g*g-v*v)/(2*g), -1.0, 1.0);
    }
    double cosine(Random& rng) const {
        if (rng.uniform() >= wd) return hgSample(gh, rng);
        double mu;
        do { mu = hgSample(gd, rng); } while (rng.uniform()*(1+alpha) > 1+alpha*mu*mu);
        return mu;
    }
    double firstMoment() const {
        return (1-wd)*gh + wd*(gd+alpha*(3*gd+2*gd*gd*gd)/5)/(1+alpha*(1+2*gd*gd)/3);
    }
};

struct Ellipsoid { Vec center, radius; };
const std::vector<Ellipsoid> tower = {
    {{0,35,0},{75,30,48}}, {{-12,85,2},{55,65,48}},
    {{0,155,0},{48,53,45}}, {{-43,125,-3},{33,37,32}},
    {{42,100,-5},{35,35,35}}, {{12,193,-5},{30,28,30}},
    {{-35,175,6},{28,29,30}}, {{0,115,-33},{30,32,23}}
};

struct Segment { double begin, end; };
struct Chord {
    std::array<Segment, 16> segments;
    int count = 0;
    double length = 0;
    double distance(double insideDistance) const {
        for (int i = 0; i < count; ++i) {
            double span = segments[i].end-segments[i].begin;
            if (insideDistance < span) return segments[i].begin+insideDistance;
            insideDistance -= span;
        }
        return INFINITY;
    }
};
Chord intersect(const std::vector<Ellipsoid>& shapes, Vec origin, Vec direction) {
    Chord result;
    for (auto shape : shapes) {
        Vec p = (origin-shape.center)/shape.radius, d = direction/shape.radius;
        double a = dot(d,d), b = dot(p,d), c = dot(p,p)-1;
        double discriminant = b*b-a*c;
        if (discriminant <= 0) continue;
        double root = std::sqrt(discriminant);
        double near = std::max(0.0, (-b-root)/a), far = (-b+root)/a;
        if (far > near) result.segments[result.count++] = {near, far};
    }
    std::sort(result.segments.begin(), result.segments.begin()+result.count,
              [](Segment a, Segment b) { return a.begin < b.begin; });
    int merged = 0;
    for (int i = 0; i < result.count; ++i) {
        auto s = result.segments[i];
        if (merged && s.begin <= result.segments[merged-1].end)
            result.segments[merged-1].end = std::max(result.segments[merged-1].end, s.end);
        else result.segments[merged++] = s;
    }
    result.count = merged;
    for (int i = 0; i < merged; ++i) result.length += result.segments[i].end-result.segments[i].begin;
    return result;
}

struct Sample { double ss[2]{}, ms[2]{}; uint64_t events = 0; };
Sample trace(Vec origin, Random& rng, const Phase& phase) {
    Sample result;
    Vec direction{0,0,1};
    Chord chord = intersect(tower, origin, direction);
    double opacity = -std::expm1(-extinction*chord.length);
    if (opacity == 0) return result;
    double distance = chord.distance(-std::log1p(-rng.uniform()*opacity)/extinction);
    double throughput = opacity;
    // E_sun = 1 on a plane normal to the disc axis; L_disc * solid_angle = 2/(1+cos(radius)).
    double diskIntegral = 2/(1+std::cos(sunRadius));
    while (std::isfinite(distance)) {
        origin = origin + direction*distance;
        throughput *= albedo;
        ++result.events;
        double mu = 1-rng.uniform()*(1-std::cos(sunRadius)), phi = 2*pi*rng.uniform();
        for (int light = 0; light < 2; ++light) {
            Vec sun = orient(light == 0 ? Vec{0,1,0} : Vec{0,0,1}, mu, phi);
            double transmittance = std::exp(-extinction*intersect(tower, origin, sun).length);
            double value = throughput*diskIntegral*phase.value(dot(direction,sun))*transmittance;
            (result.events == 1 ? result.ss[light] : result.ms[light]) += value;
        }
        // Roulette changes survival probabilities, never a fixed-order truncation.
        if (result.events >= 32) {
            if (rng.uniform() >= 0.98) break;
            throughput /= 0.98;
        }
        direction = orient(direction, phase.cosine(rng), 2*pi*rng.uniform());
        chord = intersect(tower, origin, direction);
        distance = chord.distance(-std::log(rng.uniform())/extinction);
    }
    return result;
}

struct Moments {
    double s = 0, m = 0, s2 = 0, m2 = 0, sm = 0;
    void add(double ss, double ms) { s += ss; m += ms; s2 += ss*ss; m2 += ms*ms; sm += ss*ms; }
    void write(double* out, int n) const {
        out[0] = s/n; out[1] = m/n;
        out[2] = (s2-s*s/n)/(double(n)*(n-1));
        out[3] = (m2-m*m/n)/(double(n)*(n-1));
        out[4] = (sm-s*m/n)/(double(n)*(n-1));
    }
};

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void checks(std::ostream& out, const Phase& phase) {
    std::vector<Ellipsoid> sphere{{{0,0,0},{1,1,1}}};
    Chord a = intersect(sphere,{0,0,-3},{0,0,1});
    require(std::abs(a.length-2) < 1e-13 && std::abs(a.distance(0.75)-2.75)<1e-13, "sphere chord");
    sphere.push_back({{0,0,3},{1,1,1}});
    Chord b = intersect(sphere,{0,0,-3},{0,0,1});
    require(b.count == 2 && b.length == 4 && b.distance(2.5) == 5.5, "vacuum gap");
    sphere[1].center = {0,0,1};
    require(intersect(sphere,{0,0,-3},{0,0,1}).length == 3, "overlap must not add density");
    Random rng{0x442e35b1aee19d73ULL};
    constexpr int n = 1000000;
    double sum = 0, sum2 = 0, norm = 0;
    for (int i = 0; i < n; ++i) {
        double v = phase.cosine(rng);
        sum += v; sum2 += v*v;
        double theta = (i+0.5)*pi/n;
        norm += phase.value(std::cos(theta))*std::sin(theta)*2*pi*pi/n;
    }
    double mean = sum/n, se = std::sqrt((sum2/n-mean*mean)/(n-1));
    require(std::abs(norm-1)<1e-7, "phase normalization");
    require(std::abs(mean-phase.firstMoment())<5*se, "phase sampling first moment");
    // Independent analytic delta-sun sphere L1 = albedo*p(1)*tau*exp(-tau).
    Phase hg{0.8,0,0,0};
    double tau = 4, A = -std::expm1(-tau), single = 0, single2 = 0;
    for (int i = 0; i < n; ++i) {
        double t = -std::log1p(-rng.uniform()*A);
        double v = A*0.999*hg.value(1)*std::exp(-(tau-t));
        single += v; single2 += v*v;
    }
    single /= n;
    double singleSe = std::sqrt((single2/n-single*single)/(n-1));
    double exact = 0.999*hg.value(1)*tau*std::exp(-tau);
    require(std::abs(single-exact)<5*singleSe, "analytic sphere single scattering");
    out << "{\"phase_integral\":" << norm << ",\"sample_g\":" << mean
        << ",\"exact_g\":" << phase.firstMoment() << ",\"g_standard_error\":" << se
        << ",\"sphere_L1\":" << single << ",\"sphere_L1_exact\":" << exact
        << ",\"sphere_L1_standard_error\":" << singleSe
        << ",\"chord_union_vacuum_checks\":true}";
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) { std::cerr << "reference <output> [maximum-spp=4096] [replicates=8] [threads=4]\n"; return 2; }
    int maximum = argc > 2 ? std::stoi(argv[2]) : 4096;
    int replicates = argc > 3 ? std::stoi(argv[3]) : 8;
    int threads = argc > 4 ? std::stoi(argv[4]) : 4;
    require(maximum >= 256 && replicates >= 4 && threads >= 1, "invalid sampling budget");
    std::filesystem::path output(argv[1]);
    require(std::filesystem::is_directory(output), "output directory must exist");
    std::vector<int> levels;
    for (int spp = 64; spp <= maximum; spp *= 4) levels.push_back(spp);
    if (levels.back() != maximum) levels.push_back(maximum);
    Phase phase = Phase::droplets(11.54);
    std::ostringstream checked;
    checked << std::setprecision(17);
    checks(checked, phase);
    std::cout << "checks " << checked.str() << std::endl;
    auto start = std::chrono::steady_clock::now();
    size_t pixels = width*height, stride = pixels*5;
    std::vector<double> image(2*replicates*levels.size()*stride);
    std::atomic<int> next{0}, finished{0};
    std::atomic<uint64_t> events{0}, maximumEvents{0};
    auto render = [&] {
        while (true) {
            int task = next.fetch_add(1);
            if (task >= replicates*height) break;
            int replicate = task/height, y = task%height;
            uint64_t localEvents = 0, localMaximum = 0;
            for (int x = 0; x < width; ++x) {
                size_t pixel = y*width+x;
                Random rng{0x219d681098b34102ULL + uint64_t(replicate)*0x82a2b175229d6a5bULL + pixel*0xd1342543de82ef95ULL};
                Vec origin{-100+(x+0.5)*200/width, 240-(y+0.5)*250/height, -450};
                Moments moments[2];
                size_t level = 0;
                for (int sample = 1; sample <= maximum; ++sample) {
                    Sample value = trace(origin, rng, phase);
                    localEvents += value.events;
                    localMaximum = std::max(localMaximum, value.events);
                    for (int light = 0; light < 2; ++light) moments[light].add(value.ss[light], value.ms[light]);
                    if (sample == levels[level]) {
                        for (int light = 0; light < 2; ++light)
                            moments[light].write(image.data()+((light*replicates+replicate)*levels.size()+level)*stride+pixel*5, sample);
                        ++level;
                    }
                }
            }
            events.fetch_add(localEvents);
            uint64_t old = maximumEvents.load();
            while (old < localMaximum && !maximumEvents.compare_exchange_weak(old, localMaximum)) {}
            int complete = finished.fetch_add(1)+1;
            if (complete % height == 0) std::cout << "completed_rows " << complete << '/' << replicates*height << std::endl;
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) workers.emplace_back(render);
    for (auto& worker : workers) worker.join();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    for (double v : image) require(std::isfinite(v), "nonfinite radiance or moments");
    std::ofstream binary(output/"radiance.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(image.data()), image.size()*sizeof(double));
    require(bool(binary), "radiance write failed");
    std::ofstream report(output/"reference.json");
    report << std::setprecision(17)
        << "{\n\"schema\":1,\"fixture\":\"ellipsoid-tower-v1\",\"units\":\"metres, radiance / E_sun in sr^-1\","
        << "\"shape\":[2," << replicates << ',' << levels.size() << ',' << height << ',' << width << ",5],"
        << "\"dtype\":\"little-endian float64\",\"channels\":[\"single\",\"multiple\",\"single_variance_of_mean\",\"multiple_variance_of_mean\",\"covariance_of_mean\"],"
        << "\"light_directions_to_sun\":[[0,1,0],[0,0,1]],\"sun_angular_radius_degrees\":0.266,\"sun_normal_irradiance\":1,\"sky_radiance\":0,\"ground\":false,"
        << "\"camera\":{\"type\":\"orthographic pixel centres\",\"x\":[-100,100],\"y_top_bottom\":[240,-10],\"z\":-450,\"direction\":[0,0,1]},"
        << "\"density_inside_union\":0.15,\"extinction_per_metre\":" << extinction << ",\"single_scattering_albedo\":" << albedo
        << ",\"phase\":{\"type\":\"HG+Draine\",\"diameter_um\":11.54,\"gh\":" << phase.gh << ",\"gd\":" << phase.gd << ",\"alpha\":" << phase.alpha << ",\"wd\":" << phase.wd << "},"
        << "\"roulette\":{\"starts_at_event\":32,\"survival\":0.98,\"fixed_depth_limit\":null},\"neighbors\":false,\"spp_per_replicate\":[";
    for (size_t i = 0; i < levels.size(); ++i) report << (i ? "," : "") << levels[i];
    report << "],\"ellipsoids_center_radius\":[";
    for (size_t i = 0; i < tower.size(); ++i) {
        auto e = tower[i];
        report << (i ? "," : "") << "[[" << e.center.x << ',' << e.center.y << ',' << e.center.z << "],[" << e.radius.x << ',' << e.radius.y << ',' << e.radius.z << "]]";
    }
    report << "],\"checks\":" << checked.str() << ",\"threads\":" << threads << ",\"seconds\":" << seconds
        << ",\"total_paths\":" << uint64_t(pixels)*replicates*maximum << ",\"total_events\":" << events.load() << ",\"maximum_events\":" << maximumEvents.load() << "}\n";
    require(bool(report), "report write failed");
    std::cout << "complete seconds=" << seconds << " events=" << events.load() << " maximum_events=" << maximumEvents.load() << std::endl;
}
