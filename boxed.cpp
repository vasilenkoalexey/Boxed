// MIT License

// Copyright (c) 2022 Vasilenko Alexey

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <complex>
#include <coroutine>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <ranges>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl2.h>
#include <imgui.h>
#include <implot.h>
#include <nlohmann/json.hpp>

struct coroutine {
    struct promise_type {
        coroutine get_return_object() { return {.h_ = std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        void return_void() {}
    };
    std::coroutine_handle<promise_type> h_;
};

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

template <class T, int P, typename = std::enable_if_t<std::is_floating_point<T>::value>>
T round(T a) {
    const T shift = std::pow(static_cast<T>(10.0), P);
    return std::round(a * shift) / shift;
}

template <class T, int ULP>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y) {
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ULP || std::fabs(x - y) < std::numeric_limits<T>::min();
}

template <typename... T, typename = typename std::enable_if<(true && ... && (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>)), void>::type>
auto parallel(const T... x)
{
    return 1. / (... += (1. / x));
}

static double fmin(const std::function<double(double)>& f, double& a, double b) {
    static const double ratio = (1. + std::sqrt(5.)) / 2.;
    double f1 = .0;
    double x1 = .0;
    double x2 = .0;
    while (!almost_equal<double, 2>(a, b)) {
        x1 = b - (b - a) / ratio;
        x2 = a + (b - a) / ratio;
        f1 = f(x1);
        if (f1 <= f(x2)) {
            a = x1;
        } else {
            b = x2;
        }
    }
    return f1;
}

static int
fzero(const std::function<double(double)>& f, double& b, double c, const double tolerance = std::numeric_limits<double>::epsilon()) {
    short ic = 0;

    double z = c;
    double t = b;

    if (z == t) {
        return 4;
    }

    double fb = f(t);
    int count = 1;

    if (std::fabs(fb) <= tolerance) {  // Zero at b
        return 2;
    }

    z = t + 0.5 * (z - t);

    double fz = f(z);
    double fc = fz;
    count = 2;

    const auto sign = [](double y) { return ((y < 0) ? -1 : 1); };

    if (sign(fz) == sign(fb)) {
        t = c;
        fc = f(t);
        count = 3;

        if (std::fabs(fc) <= tolerance) {  // Zero at c
            b = t;
            return 2;
        }
        if (sign(fz) != sign(fc)) {
            b = z;
            fb = fz;
        } else {
            return 4;
        }
    } else {
        c = z;
    }

    double a = c;
    double fa = fc;
    double acbs = fabs(a - b);

    double fx = fabs(fb);
    if (fx < fabs(fc)) {
        fx = fabs(fc);
    }

    do {
        // Arrange so fabs(f(b)) LE fabs(f(c))
        if (fabs(fc) < fabs(fb)) {  // Interchange if necessary
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = a;
            fc = fa;
        }

        const double cmb = 0.5 * (c - b);
        const double acmb = fabs(cmb);
        const double tol = 2. * tolerance * fabs(b) + tolerance;

        if (acmb <= tol) {
            break;
        }

        if (fb == .0) {
            return 2;
        }

        if (count >= 100) {
            return 5;
        }

        /*Calculate new iterate implicitly as b + p/q, where p is arranged to be
        >= 0. This implicit form is used to prevent overflow.*/

        double p = (b - a) * fb;
        double q = fa - fb;

        if (p < 0) {
            p = -p;
            q = -q;
        }

        /*Update a and check for satisfactory reduction in the size of the
        bracketing interval. If not, perform bisection.*/

        a = b;
        fa = fb;
        ++ic;

        if ((ic >= 4) && (8 * acmb >= acbs)) {
            b = 0.5 * (c + b);
        } else {
            if (ic >= 4) {
                ic = 0;
                acbs = acmb;
            }
            if (p <= tol * fabs(q)) {  // Test for too small a change
                b += tol * sign(cmb);
            } else {  // Root between b and (b + c)/2
                if (p < cmb * q) {
                    b += p / q;  // Use secant rule
                } else {
                    b = 0.5 * (c + b);
                }
            }
        }  // End else !((ic >= 4) && (8 * acmb >= acbs))

        // Have now computed new iterate, b.
        fb = f(b);
        count += 1;

        // Decide if next step interpolation or extrapolation.

        if (sign(fb) == sign(fc)) {
            c = a;
            fc = fa;
        }

    } while (count < 100);

    if (sign(fb) == sign(fc)) {
        return 4;
    }

    if (fabs(fb) > fx) {
        return 3;
    }

    return 1;
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Boxed", NULL, NULL);
    if (window == NULL) {
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // Disable vsync

    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    struct driver_model {
        std::string model;
        double mms;
        double cms;
        double re;
        double bl;
        double rms;
        double sd;
        double le;

        double fs;
        double qes;
        double qms;
        double qts;
        double vas;

        double xmax;
        double xmech;
        double pe;

        double depth;
        double mdepth;
        double mdia;
        double vcd;
        bool operator==(const driver_model&) const = default;
    };

    constexpr int id0 = 14666;
    constexpr int id1 = 14667;
    constexpr int id2 = 14668;

    static double rg = 0.01; /* Output resistance of source or amplifier */
    static double ql = 7.;
    static double vb;         /* Box Volume */
    static double fb;         /* Box Tuning Frequency */
    static double sp = .0001; /* Port Area */
    static const double sp_min = .0001;
    static double k = 0.732;   /* End Correction factor */
    static double rh0 = 1.184; /* Dencity of air */
    static double c = 346.1;   /* Velocity of sound in air */
    static double pe = .1;

    static const double f64_zero = 0.;

    constexpr int nfreq = 400;
    constexpr double low = 10.;
    constexpr double high = 1000.;
    static std::array<double, nfreq> freq;
    static const double gap = std::exp((std::log(high) - std::log(low)) / (nfreq));
    for (double fp = low; double& f : freq) {
        f = fp;
        fp *= gap;
    }

    static std::vector<std::pair<std::string, std::vector<driver_model>>> vendors;

    auto load = [&]() -> coroutine {
        for (const auto& entry : std::filesystem::directory_iterator{"./drivers"}) {
            if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".json") {
                nlohmann::json json;
                try {
                    std::ifstream(entry.path().string()) >> json;
                } catch (nlohmann::json::parse_error) {
                    continue;
                }
                std::string vendor{};
                driver_model driver{};
                for (const auto& [key, value] : json.items()) {
                    if (value.is_string()) {
                        if (key == "vendor")
                            vendor = value.get<std::string>();
                        else if (key == "model")
                            driver.model = value.get<std::string>();
                    } else if (value.is_number()) {
                        if (key == "mms") {
                            driver.mms = value.get<double>();
                        } else if (key == "cms") {
                            driver.cms = value.get<double>();
                        } else if (key == "re") {
                            driver.re = value.get<double>();
                        } else if (key == "bl") {
                            driver.bl = value.get<double>();
                        } else if (key == "rms") {
                            driver.rms = value.get<double>();
                        } else if (key == "sd") {
                            driver.sd = value.get<double>();
                        } else if (key == "le") {
                            driver.le = value.get<double>();
                        } else if (key == "fs") {
                            driver.fs = value.get<double>();
                        } else if (key == "qes") {
                            driver.qes = value.get<double>();
                        } else if (key == "qms") {
                            driver.qms = value.get<double>();
                        } else if (key == "qts") {
                            driver.qts = value.get<double>();
                        } else if (key == "vas") {
                            driver.vas = value.get<double>();
                        } else if (key == "xmax") {
                            driver.xmax = value.get<double>();
                        } else if (key == "xmech") {
                            driver.xmech = value.get<double>();
                        } else if (key == "pe") {
                            driver.pe = value.get<double>();
                        } else if (key == "depth") {
                            driver.depth = value.get<double>();
                        } else if (key == "mdepth") {
                            driver.mdepth = value.get<double>();
                        } else if (key == "mdia") {
                            driver.mdia = value.get<double>();
                        } else if (key == "vcd") {
                            driver.vcd = value.get<double>();
                        }
                    }
                }
                if (driver.cms == .0 && driver.fs != .0 && driver.mms != .0) {
                    driver.cms = std::pow(1. / (driver.fs * 2. * std::numbers::pi), 2.) / driver.mms;
                }
                if (driver.rms == .0 && driver.qms != .0) {
                    driver.rms = std::sqrt(driver.mms / driver.cms) / driver.qms;
                }
                if (driver.sd == .0 || driver.cms == .0 || driver.mms == .0 || driver.bl == .0 || driver.re == .0) {
                    continue;
                }
                if (!vendor.empty() && !driver.model.empty()) {
                    if (const auto drivers = std::ranges::find_if(vendors, [&vendor](const auto& e) { return e.first == vendor; }); drivers != vendors.end()) {
                        drivers->second.emplace_back(driver);
                    }
                    else {
                        vendors.emplace_back(std::make_pair(vendor, std::vector<driver_model>{driver}));
                    }
                    co_await std::suspend_always{};
                }
            }
        }
    };

    auto coroutine_handle = load().h_;
    auto& promise = coroutine_handle.promise();

    while (!glfwWindowShouldClose(window)) {
        if (coroutine_handle) {
            if (coroutine_handle.done()) {
                coroutine_handle.destroy();
                coroutine_handle = nullptr;
            } else {
                coroutine_handle();
            }
        }

        glfwPollEvents();

        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        static bool show_driver = true;
        static bool show_design = true;
        static bool show_options = true;
        static bool show_сrosshairs = true;

        static driver_model driver{};

        static int index_vendor = 0;
        static int index_model = 0;
        static int index_vendor_prev = -1;
        static int index_model_prev = -1;

        if (show_driver) {
            ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, FLT_MAX));
            if (ImGui::Begin("Driver database", &show_driver, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
                ImGui::PushItemWidth(220);
                ImGui::BeginGroup();

                ImGui::Combo(
                    "Vendor", &index_vendor, [](void* map, int idx, const char** out_text) {
                        *out_text = static_cast<std::vector<std::pair<std::string, std::vector<driver_model>>>*>(map)->at(idx).first.c_str();
                        return true;
                    },
                    static_cast<void*>(&vendors), static_cast<int>(vendors.size()), 25);

                if (index_vendor_prev != index_vendor) {
                    index_model = 0;
                }

                const auto vendor = &vendors.at(index_vendor).second;
                ImGui::Combo(
                    "Model", &index_model, [](void* vector, int idx, const char** out_text) {
                        *out_text = static_cast<std::vector<driver_model>*>(vector)->at(idx).model.c_str();
                        return true;
                    },
                    static_cast<void*>(vendor), static_cast<int>(vendor->size()), 25);

                if (index_vendor_prev != index_vendor || index_model_prev != index_model) {
                    driver = vendor->at(index_model);
                    ImPlot::SetNextAxesToFit();
                }

                ImGui::EndGroup();

                ImGui::SameLine();

                ImGui::BeginGroup();
                if (ImGui::Button("Reload", ImVec2(-1.f, 0.f))) {
                    vendors.clear();
                    if (coroutine_handle)
                        coroutine_handle.destroy();
                    coroutine_handle = load().h_;
                    promise = coroutine_handle.promise();

                    index_vendor_prev = -1;
                    index_model_prev = -1;
                    index_vendor = 0;
                    index_model = 0;

                    driver = vendors.front().second.front();
                    ImPlot::SetNextAxesToFit();
                }

                const bool driver_modified = (!vendor->empty() && driver != vendor->at(index_model));

                if (!driver_modified) {
                    ImGui::BeginDisabled();
                }
                const ImVec2 size = ImVec2(50.f, 0.f);
                if (ImGui::Button("Save", size)) {
                    vendor->at(index_model) = driver;
                    nlohmann::json json;
                    json["vendor"] = vendors.at(index_vendor).first;
                    json["model"] = driver.model;
                    json["mms"] = round<double, 4>(driver.mms);
                    json["cms"] = round<double, 7>(driver.cms);
                    json["re"] = round<double, 3>(driver.re);
                    json["bl"] = round<double, 5>(driver.bl);
                    if (driver.rms > .0) {
                        json["rms"] = round<double, 5>(driver.rms);
                    }
                    json["sd"] = round<double, 5>(driver.sd);
                    json["le"] = round<double, 6>(driver.le);
                    if (driver.fs > .0) {
                        json["fs"] = round<double, 3>(driver.fs);
                    }
                    if (driver.qes > .0) {
                        json["qes"] = round<double, 3>(driver.qes);
                    }
                    if (driver.qms > .0) {
                        json["qms"] = round<double, 3>(driver.qms);
                    }
                    if (driver.qts > .0) {
                        json["qts"] = round<double, 3>(driver.qts);
                    }
                    json["vas"] = round<double, 5>(driver.vas);
                    if (driver.xmax > .0) {
                        json["xmax"] = round<double, 4>(driver.xmax);
                    }
                    if (driver.xmech > .0) {
                        json["xmech"] = round<double, 4>(driver.xmech);
                    }
                    if (driver.pe > .0) {
                        json["pe"] = round<double, 1>(driver.pe);
                    }
                    if (driver.depth > .0) {
                        json["depth"] = round<double, 1>(driver.depth);
                    }
                    if (driver.mdepth > .0) {
                        json["mdepth"] = round<double, 1>(driver.mdepth);
                    }
                    if (driver.mdia > .0) {
                        json["mdia"] = round<double, 1>(driver.mdia);
                    }
                    if (driver.vcd > .0) {
                        json["vcd"] = round<double, 1>(driver.vcd);
                    }
                    std::ofstream o(vendors.at(index_vendor).first + " " + driver.model + ".json");
                    o << std::setw(4) << json << std::endl;
                }

                ImGui::SameLine();
                if (ImGui::Button("Reset", size)) {
                    driver = vendor->at(index_model);
                }

                if (!driver_modified) {
                    ImGui::EndDisabled();
                }

                ImGui::EndGroup();

                ImGui::PopItemWidth();

                ImGui::PushItemWidth(135);

                if (ImGui::CollapsingHeader("Fundamental parameters")) {
                    static const double sd_min = .00001;
                    ImGui::DragScalar("Surface Area Of Cone", ImGuiDataType_Double, &driver.sd, .00001f, &sd_min, nullptr, "Sd: %.5f m2", ImGuiSliderFlags_AlwaysClamp);
                    static const double mms_min = .0001;
                    ImGui::DragScalar("Diaphragm Mass Including Air-Load", ImGuiDataType_Double, &driver.mms, .0001f, &mms_min, nullptr, "Mms: %.4f kg", ImGuiSliderFlags_AlwaysClamp);
                    static const double cms_min = .0000001;
                    ImGui::DragScalar("Mechanical Compliance Of Suspension", ImGuiDataType_Double, &driver.cms, .0000001f, &cms_min, nullptr, "Cms: %.7f m/N", ImGuiSliderFlags_AlwaysClamp);
                    static const double rms_min = .00001;
                    ImGui::DragScalar("Mechanical Resistance", ImGuiDataType_Double, &driver.rms, .00001f, &rms_min, nullptr, "Rms: %.5f kg/s", ImGuiSliderFlags_AlwaysClamp);
                    static const double le_min = .0;
                    ImGui::DragScalar("Voice coil inductance", ImGuiDataType_Double, &driver.le, .000001f, &le_min, nullptr, "Le: %.6f H", ImGuiSliderFlags_AlwaysClamp);
                    static const double re_min = .001;
                    ImGui::DragScalar("DC resistance of the voice coil", ImGuiDataType_Double, &driver.re, .001f, &re_min, nullptr, "Re: %.3f ohms", ImGuiSliderFlags_AlwaysClamp);
                    static const double bl_min = .1;
                    ImGui::DragScalar("BL Product", ImGuiDataType_Double, &driver.bl, .00001f, &bl_min, nullptr, "BL: %.5f Tm", ImGuiSliderFlags_AlwaysClamp);
                }

                if (ImGui::CollapsingHeader("Small signal parameters")) {
                    static const double fs_min = 10.;
                    ImGui::DragScalar("Resonant Frequency", ImGuiDataType_Double, &driver.fs, .001f, &fs_min, nullptr, "Fs: %.3f Hz", ImGuiSliderFlags_AlwaysClamp);
                    double fs = 1. / (2. * std::numbers::pi * std::sqrt(driver.cms * driver.mms));
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Resonant Frequency'", ImGuiDataType_Double, &fs, .0f, nullptr, nullptr, "Fs': %.3f Hz");
                    ImGui::EndDisabled();
                    static const double qms_min = .001;
                    ImGui::DragScalar("Mechanical Q", ImGuiDataType_Double, &driver.qms, .001f, &qms_min, nullptr, "Qms: %.3f", ImGuiSliderFlags_AlwaysClamp);
                    double qms = std::sqrt(driver.mms / driver.cms) / driver.rms;
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Mechanical Q'", ImGuiDataType_Double, &qms, .0f, nullptr, nullptr, "Qms': %.3f");
                    ImGui::EndDisabled();
                    static const double qes_min = .001;
                    ImGui::DragScalar("Electrical Q", ImGuiDataType_Double, &driver.qes, .001f, &qes_min, nullptr, "Qes: %.3f", ImGuiSliderFlags_AlwaysClamp);
                    double qes = std::sqrt(driver.mms / driver.cms) * driver.re / std::pow(driver.bl, 2);
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Electrical Q'", ImGuiDataType_Double, &qes, .0f, nullptr, nullptr, "Qes': %.3f");
                    ImGui::EndDisabled();
                    static const double qts_min = .001;
                    ImGui::DragScalar("Total Q", ImGuiDataType_Double, &driver.qts, .001f, &qts_min, nullptr, "Qts: %.3f", ImGuiSliderFlags_AlwaysClamp);
                    double qts = qms * qes / (qms + qes);
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Total Q'", ImGuiDataType_Double, &qts, .0f, nullptr, nullptr, "Qts': %.3f");
                    ImGui::EndDisabled();
                    static const double vas_min = .00001;
                    ImGui::DragScalar("Equivalent Compliance Volume", ImGuiDataType_Double, &driver.vas, .00001f, &vas_min, nullptr, "Vas: %.5f m3", ImGuiSliderFlags_AlwaysClamp);
                    double vas = rh0 * std::pow(c, 2.) * std::pow(driver.sd, 2.) * driver.cms;
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Equivalent Compliance Volume'", ImGuiDataType_Double, &vas, .0f, nullptr, nullptr, "Vas': %.5f m3");
                    ImGui::EndDisabled();
                }

                if (ImGui::CollapsingHeader("Large signal parameters")) {
                    static const double xmax_min = .0;
                    ImGui::DragScalar("Maximum Linear Excursion", ImGuiDataType_Double, &driver.xmax, 0.0001f, &xmax_min, nullptr, "Xmax: %.4f m", ImGuiSliderFlags_AlwaysClamp);
                    static const double xmech_min = .0;
                    ImGui::DragScalar("Maximum Mechanical Excursion", ImGuiDataType_Double, &driver.xmech, 0.0001f, &xmech_min, nullptr, "Xmech: %.4f m", ImGuiSliderFlags_AlwaysClamp);
                    static const double pe_min = 1.;
                    ImGui::DragScalar("Thermal Power Handling", ImGuiDataType_Double, &driver.pe, 0.1f, &pe_min, nullptr, "Pe: %.1f W", ImGuiSliderFlags_AlwaysClamp);
                }

                if (ImGui::CollapsingHeader("Dimensions")) {
                    ImGui::DragScalar("Depth", ImGuiDataType_Double, &driver.depth, 0.1f, &f64_zero, nullptr, "%.1f mm", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::DragScalar("Magnet Depth", ImGuiDataType_Double, &driver.mdepth, 0.1f, &f64_zero, nullptr, "%.1f mm", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::DragScalar("Magnet Diameter", ImGuiDataType_Double, &driver.mdia, 0.1f, &f64_zero, nullptr, "%.1f mm", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::DragScalar("Voice Coil Diameter", ImGuiDataType_Double, &driver.vcd, 0.1f, &f64_zero, nullptr, "%.1f mm", ImGuiSliderFlags_AlwaysClamp);
                    /* magnet volume */
                    const double mv = std::numbers::pi * std::pow(driver.mdia / 2., 2.) * driver.mdepth;
                    /* surface area radius */
                    const double sr = driver.sd / 2.;
                    /* voice-coil radius */
                    const double vcr = driver.vcd / 2.;
                    /* cone height */
                    const double ch = std::max(driver.depth - driver.mdepth - driver.xmax, 0.);
                    /* cone volume */
                    const double cv = std::numbers::pi * ch * (std::pow(vcr, 2.) + std::pow(sr, 2.) + vcr * sr) / 3;
                    /* total volume */
                    static double vol = .0;
                    vol = (mv + cv) / 1e+6;
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Driver volume", ImGuiDataType_Double, &vol, .0f, nullptr, nullptr, "%.3f L");
                    ImGui::EndDisabled();
                }
                ImGui::PopItemWidth();
                ImGui::End();
            }
        }
        
        static int voice_coil_connection = 0;
        static int driver_count = 1;
        
        if (driver_count < 1) {
            driver_count = 1;
        }        

        const double pe_max = driver.pe * driver_count;

        if (index_vendor_prev != index_vendor || index_model_prev != index_model) {
            pe = pe_max;
            const double fs = (driver.fs > .0) ? driver.fs : 1. / (2. * std::numbers::pi * std::sqrt(driver.cms * driver.mms));
            const double vas = (driver.vas > .0) ? driver.vas : rh0 * std::pow(c, 2.) * std::pow(driver.sd, 2.) * driver.cms;
            const double qes = (driver.qes > .0) ? driver.qes : std::sqrt(driver.mms / driver.cms) * driver.re / std::pow(driver.bl, 2.);
            const double qms = (driver.qms > .0) ? driver.qms : std::sqrt(driver.mms / driver.cms) / driver.rms;
            const double qts = (driver.qts > .0) ? driver.qts : qms * qes / (qms + qes);
            vb = vas * std::pow(qts / 0.4, 3.) / 1.1;
            fb = fs * 0.42 / qts;
        } else if(pe > pe_max) {
            pe = pe_max;
        }

        double re = (voice_coil_connection == 0) ? driver.re / driver_count : driver.re * driver_count;
        const double le = (voice_coil_connection == 0) ? driver.le / driver_count : driver.le * driver_count;
        const double sd = driver.sd * driver_count;        
        const double mms = driver.mms * driver_count;
        const double vas = driver.vas * driver_count;

        const double sd2 = std::pow(sd, 2.);
        const double bl2 = std::pow(driver.bl, 2.);

        /* Cab - Acoustic compliance of air in enclosure */
        const double cab = vb / (rh0 * c * c);
        /* Cas - acoustic compliance of driver suspension */
        const double cas = (vas > .0) ? vas / (rh0 * c * c) : sd2 * driver.cms;
        /* Wb */
        const double wb = 2. * std::numbers::pi * fb;
        /* Cmes - electrical capacitance due to driver mass */
        const double cmes = mms / bl2;
        /* Lces - electrical inductance due to driver compliance */
        const double lces = driver.cms * bl2;
        /* Ras - acoustic resistance of driver suspension losses */
        const double ras = driver.rms / sd2;
        /* Res - electrical resistance due to driver suspension losses */
        const std::complex res = bl2 / driver.rms;
        /* Lceb */
        const double lceb = bl2 * cab / sd2;
        /* Ral - acoustic resistance of enclosure lossescaused by leakage */
        const std::complex ral = ql / (wb * cab);
        /* Rel - acoustic resistance of enclosure lossescaused by leakage */
        const std::complex rel = bl2 / (sd2 * ral);
        /* Mas - acoustic mass of driver diaphragm assembly including air load */
        const double mas = mms / sd2;
        /* eg - Open-circuit output voltage of source */
        const double eg = std::sqrt(pe * re);
        /* Pg - acoustic driving pressure */
        const double pg = eg * driver.bl / ((rg + re) * sd);

        const double eg_spl = std::sqrt(re);
        const double pg_spl = eg_spl * driver.bl / ((rg + re) * sd);

        /* Rat - acoustic resistance of total driver-circuit losses */
        const double rat = ras + bl2 / (sd2 * (rg + re));

        const auto calc_lpa = [&]() {
            return std::pow(c, 2.) * sp / (std::pow(2. * std::numbers::pi, 2.) * std::pow(fb, 2.) * vb);
        };

        std::function<void(const double, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const)> calc_vented_box;
        calc_vented_box = [&](const double f, double* const _g, double* const _phg, double* const _zvc, double* const _phzvc, double* const _gd, double* const _x, double* const _v, double* const _spld, double* const _splp, double* const _spl) {
            const auto phase = [](const std::complex<double>& arg) {
                const auto degrees = [](const double radians) {
                    return (360. / (2. * std::numbers::pi)) * radians;
                };
                if (arg.real() == .0) {
                    if (arg.imag() == .0) {
                        return .0;
                    }
                    if (arg.imag() > .0) {
                        return 90.;
                    }
                    return -90.0;
                }
                if (arg.real() < .0) {
                    if (arg.imag() == .0) {
                        return 180.;
                    }
                    if (arg.imag() > .0) {
                        return 180. + degrees(std::atan(arg.imag() / arg.real()));
                    }
                    return -180. + degrees(std::atan(arg.imag() / arg.real()));
                }
                return degrees(std::atan(arg.imag() / arg.real()));
            };

            /* Map - acoustic mass of port including air load */
            const double map = calc_lpa() * rh0 / sp;
            /* Cmep */
            const double cmep = cab * map / lceb;

            const std::complex s(0., f * 2. * std::numbers::pi);
            const std::complex zaa = s * map;
            const std::complex zas = rat + s * mas + 1. / (s * cas);
            const std::complex zab = 1. / (s * cab);

            const std::complex<double> ralc = ral;

            const std::complex z0 = parallel(zab, ral);

            const std::complex up_spl = pg_spl * z0 / (zaa * z0 + zas * (zaa + z0));

            const std::complex z1 = zas + parallel(zaa, zab, ral);
            const std::complex z2 = parallel(zaa, ral);

            /* U0 - total volume velocity leaving enclosure boundaries */
            const std::complex u0 = pg * z2 / (zab * z2 + zas * (zab + z2));

            const std::complex ud_spl = pg_spl / z1;
            if (_x != nullptr) {
                /* Volume velocity of driver diaphragm */
                const std::complex ud = pg / z1;
                /* Diaphragm dsiplacement */
                *_x = std::sqrt(2.) * std::abs(ud / (sd * s));
            }
            const std::complex g = s * mas * u0 / pg;
            if (_g != nullptr) {
                /* Response */
                *_g = 20. * std::log10(std::abs(g));
            }
            const double phg = phase(g);
            if (_phg != nullptr) {
                /* Response phase */
                *_phg = phg;
            }
            if (_gd != nullptr) {
                const double f_prev = f / gap;
                const std::complex s_prev(0., f_prev * 2. * std::numbers::pi);
                const std::complex zas_prev = rat + s_prev * mas + 1. / (s_prev * cas);
                const std::complex zaa_prev = s_prev * map;
                const std::complex zab_prev = 1. / (s_prev * cab);

                const std::complex z3 = parallel(zaa_prev, ral);

                const std::complex u0_prev = pg * z3 / (zab_prev * z3 + zas_prev * (zab_prev + z3));
                const std::complex g_prev = s_prev * mas * u0_prev / pg;
                /* Group delay */
                if (const double phg_prev = phase(g_prev); (phg > .0 && phg_prev > .0) || (phg < .0 && phg_prev < .0)) {
                    *_gd = (-1. / 360.) * ((phg - phg_prev) / (f - f_prev));
                }
                else {
                    double gd_prev;
                    calc_vented_box(f_prev, nullptr, nullptr, nullptr, nullptr, &gd_prev, nullptr, nullptr, nullptr, nullptr, nullptr);
                    double gd_next;
                    calc_vented_box(f * gap, nullptr, nullptr, nullptr, nullptr, &gd_next, nullptr, nullptr, nullptr, nullptr, nullptr);
                    *_gd = .5 * (gd_prev + gd_next);
                }
            }
            if (_v != nullptr) {
                /* Up - volume velocity of port */
                const std::complex up = pg * z0 / (zaa * z0 + zas * (zaa + z0));
                /* Port air velocity */
                *_v = std::sqrt(2.) * std::abs(up) / sp;
            }
            if (_spld != nullptr) {
                /* Driver sound pressure level */
                *_spld = 79.6 + 20. * std::log10(std::abs(ud_spl * rh0 * s));
            }
            if (_splp != nullptr) {
                /* Port sound pressure level */
                *_splp = 79.6 + 20. * std::log10(std::abs(up_spl * rh0 * s));
            }
            if (_spl != nullptr) {
                /* Overall sound pressure level */
                *_spl = 79.6 + 20. * std::log10(std::abs((up_spl - ud_spl) * s * rh0));
            }
            if (_zvc != nullptr) {
                /* Voice-coil impedance */
                const std::complex ze = rel + s * lceb + 1. / (s * cmep);
                const std::complex zs = parallel(res, s * lces, 1. / (s * cmes));
                const std::complex zm = parallel(ze, zs);
                const std::complex zvc = re + s * le + zm;
                *_zvc = std::abs(zvc);
                /* And voice-coil impedance phase */
                if (_phzvc != nullptr) {
                    *_phzvc = phase(zvc);
                }
            } else if (_phzvc != nullptr) {
                /* Voice-coil impedance phase only */
                const std::complex ze = rel + s * lceb + 1. / (s * cmep);
                const std::complex zs = parallel(res, s * lces, 1. / (s * cmes));
                const std::complex zm = parallel(ze, zs);
                const std::complex zvc = re + s * le + zm;
                *_phzvc = phase(zvc);
            }
        };

        const auto calc_v = [&](const double f) {
            double v;
            calc_vented_box(f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &v, nullptr, nullptr, nullptr);
            return v;
        };

        std::vector<double> g(nfreq), phg(nfreq), zvc(nfreq), phzvc(nfreq), gd(nfreq), x(nfreq), v(nfreq), spld(nfreq), splp(nfreq), spl(nfreq);

        if (index_vendor_prev != index_vendor || index_model_prev != index_model) {
            const double sp_prev = sp;
            const auto f = [&](const double arg) {
                sp = arg;
                double a = freq.front();
                return fmin([&](const double f) { return calc_v(f); }, a, freq.back()) - 9.;
            };
            double b = sp_min;
            const int r = fzero(f, b, 2 * driver.sd);
            if (r == 5) {
                sp = sp_prev;
            }
        }

        index_vendor_prev = index_vendor;
        index_model_prev = index_model;

        auto it_g = g.begin();
        auto it_phg = phg.begin();
        auto it_zvc = zvc.begin();
        auto it_phzvc = phzvc.begin();
        auto it_gd = gd.begin();
        auto it_x = x.begin();
        auto it_v = v.begin();
        auto it_spld = spld.begin();
        auto it_splp = splp.begin();
        auto it_spl = spl.begin();

        for (const auto f : freq)
            calc_vented_box(f, &*it_g++, &*it_phg++, &*it_zvc++, &*it_phzvc++, &*it_gd++, &*it_x++, &*it_v++, &*it_spld++, &*it_splp++, &*it_spl++);

        if (show_design) {
            ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, FLT_MAX));
            if (ImGui::Begin("Enclosure design", &show_design, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
                ImGui::PushItemWidth(180);

                if (ImGui::CollapsingHeader("Driver", ImGuiTreeNodeFlags_SpanFullWidth)) {
                    static const double pe_min = .1;
                    ImGui::InputInt("Count", &driver_count);
                    ImGui::Combo("Voice coil connection", &voice_coil_connection, "Parallel\0Serial\0\0");
                    ImGui::DragScalar("Input power", ImGuiDataType_Double, &pe, .1f, &pe_min, &pe_max, "pe: %.1f W", ImGuiSliderFlags_AlwaysClamp);                                        
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Total Re", ImGuiDataType_Double, &re, .0f, nullptr, nullptr, "Re: %.3f Ohm");
                    ImGui::EndDisabled();
                }

                if (ImGui::CollapsingHeader("Box", ImGuiTreeNodeFlags_SpanFullWidth)) {
                    static const double vbl_min = 1.;
                    double vbl = 1000. * vb;
                    ImGui::DragScalar("Volume", ImGuiDataType_Double, &vbl, .01f, &vbl_min, nullptr, "Vb %.2f L", ImGuiSliderFlags_AlwaysClamp);
                    vb = vbl / 1000.;

                    static const double fb_min = 10.;
                    ImGui::DragScalar("Tuning Frequency", ImGuiDataType_Double, &fb, 0.01f, &fb_min, nullptr, "Fb %.2f Hz", ImGuiSliderFlags_AlwaysClamp);

                    static const double ql_max = 30.;
                    static const double ql_min = 3.;
                    ImGui::SliderScalar("Leakage Q", ImGuiDataType_Double, &ql, &ql_min, &ql_max, "Ql %.f", ImGuiSliderFlags_AlwaysClamp);
                }

                if (ImGui::CollapsingHeader("Port")) {
                    ImGui::DragScalar("Area", ImGuiDataType_Double, &sp, 0.00001f, &sp_min, nullptr, "Sp %.5f m2", ImGuiSliderFlags_AlwaysClamp);

                    double dp = 2. * std::sqrt(sp / std::numbers::pi);
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Diameter", ImGuiDataType_Double, &dp, .0f, nullptr, nullptr, "Dp %.4f m");

                    static double hp = .01;
                    double wp = sp / hp;
                    if (wp < 0.01) {
                        wp = 0.01;
                    }
                    ImGui::DragScalar("Width", ImGuiDataType_Double, &wp, .0f, nullptr, nullptr, "Wp %.3f m");
                    ImGui::EndDisabled();

                    const double hp_max = std::sqrt(sp);
                    if (hp > hp_max) {
                        hp = hp_max;
                    }
                    static const double hp_min = .01;
                    ImGui::SliderScalar("Height", ImGuiDataType_Double, &hp, &hp_min, &hp_max, "Hp %.3f m", ImGuiSliderFlags_AlwaysClamp);

                    static const double k_min = 0.;
                    ImGui::DragScalar("End correction", ImGuiDataType_Double, &k, .0001f, &k_min, nullptr, "k %.3f");

                    double lpa = calc_lpa();
                    double lpm = lpa - k * dp;
                    if (lpm < .0) {
                        lpm = .0;
                    }
                    ImGui::BeginDisabled();
                    ImGui::DragScalar("Length", ImGuiDataType_Double, &lpm, .0f, nullptr, nullptr, "Lp %.3f m");
                    ImGui::DragScalar("Length acoustical", ImGuiDataType_Double, &lpa, .0f, nullptr, nullptr, "Lpa %.3f m");
                    ImGui::EndDisabled();
                }
                ImGui::PopItemWidth();
                ImGui::End();
            }
        }

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        if (ImGui::Begin("Plot", NULL, ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings)) {
            constexpr ImPlotFlags flagsPlot = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;
            ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32_BLACK_TRANS);

            if (ImGui::BeginTabBar("Plots")) {
                if (ImGui::BeginTabItem("Response")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Response [dB]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(g);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), g.data(), nfreq);
                        const auto calc_g = [&](const double f) {
                            double g;
                            calc_vented_box(f, &g, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                            return g;
                        };
                        static bool show_f3 = true;
                        if (show_f3) {
                            double a = freq.front();
                            const int r = fzero([&](const double v) { return calc_g(v) + 3.; }, a, freq.back());
                            if (r != 5) {
                                ImPlot::DragLineX(id0, &a, ImVec4(.15f, .8f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                                const ImPlotRect rect = ImPlot::GetPlotLimits();
                                ImPlot::Annotation(a, rect.Min().y, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Response: %.2f dB", a, calc_g(a));
                            }
                        }
                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id1, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double g;
                            calc_vented_box(f, &g, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id2, &g, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, g, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Response: %.2f dB", f, g);
                        }
                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::MenuItem("F3", NULL, &show_f3);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Response phase")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        std::vector<double> l0;
                        std::vector<double> l1;
                        for (const double i : phg) {
                            if (i < 0) {
                                l0.push_back(i);
                            } else {
                                l1.push_back(i);
                            }
                        }
                        ImPlot::SetupAxes("Frequency [Hz]", "Response phase [deg]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(phg);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), l0.data(), static_cast<int>(l0.size()));
                        ImPlot::PlotLine("##line", freq.data() + l0.size(), l1.data(), static_cast<int>(l1.size()));
                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double phg;
                            calc_vented_box(f, nullptr, &phg, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id1, &phg, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, phg, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Response phase: %.2f deg", f, phg);
                        }
                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Group delay")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Group delay [sec]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(gd);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), gd.data(), nfreq);
                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double gd;
                            calc_vented_box(f, nullptr, nullptr, nullptr, nullptr, &gd, nullptr, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id1, &gd, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, gd, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Group delay: %.4f sec", f, gd);
                        }
                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Cone excursion")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Cone excursion [m]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(x);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), x.data(), nfreq);
                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double x;
                            calc_vented_box(f, nullptr, nullptr, nullptr, nullptr, nullptr, &x, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id1, &x, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, x, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Cone excursion: %.4f m", f, x);
                        }
                        static bool show_xmax = true;
                        static bool show_xmech = true;
                        if (driver.xmax > .0 && show_xmax) {
                            ImPlot::DragLineY(id2, &driver.xmax, ImVec4(.15f, .9f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            const ImPlotRect rect = ImPlot::GetPlotLimits();
                            ImPlot::Annotation(rect.Max().x, driver.xmax, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Xmax %.4f m", driver.xmax);
                        }
                        if (driver.xmech > .0 && show_xmech) {
                            ImPlot::DragLineY(id2, &driver.xmech, ImVec4(.15f, .9f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            const ImPlotRect rect = ImPlot::GetPlotLimits();
                            ImPlot::Annotation(rect.Max().x, driver.xmech, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Xmech %.4f m", driver.xmech);
                        }
                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            if (driver.xmax > .0) {
                                ImGui::MenuItem("Xmax", NULL, &show_xmax);
                            }
                            if (driver.xmech > .0) {
                                ImGui::MenuItem("Xmech", NULL, &show_xmech);
                            }
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Port air velocity")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Port air velocity [m/sec]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(v);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), v.data(), nfreq);
                        static bool show_max_v = true;
                        if (show_max_v) {
                            double a = freq.front();
                            double max_v = fmin([&](const double arg) { return calc_v(arg); }, a, freq.back());
                            ImPlot::DragLineX(id0, &a, ImVec4(.15f, .8f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            const ImPlotRect rect = ImPlot::GetPlotLimits();
                            ImPlot::Annotation(a, rect.Min().y, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Port air velocity: %.2f m/sec", a, max_v);
                        }

                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id1, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double v;
                            calc_vented_box(f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &v, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id2, &v, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, v, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Port air velocity: %.2f m/sec", f, v);
                        }

                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::MenuItem("Max velocity", NULL, &show_max_v);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Voice-coil impedance magnitude")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Voice-coil impedance magnitude [ohms]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(zvc);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), zvc.data(), nfreq);

                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double zvc;
                            calc_vented_box(f, nullptr, nullptr, &zvc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id1, &zvc, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, zvc, ImVec4(0.15f, 0.15f, 0.15f, 1), ImVec2(5, -5), true,
                                               "Freq: %.2f Hz, Voice-coil "
                                               "impedance magnitude: %.2f ohms",
                                               f, zvc);
                        }

                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Voice-coil impedance phase")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Voice-coil impedance phase [deg]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(phzvc);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("##line", freq.data(), phzvc.data(), nfreq);

                        if (show_сrosshairs && ImPlot::IsPlotHovered()) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double phzvc;
                            calc_vented_box(f, nullptr, nullptr, nullptr, &phzvc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                            ImPlot::DragLineY(id1, &phzvc, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            ImPlot::Annotation(f, phzvc, ImVec4(0.15f, .15f, .15f, 1), ImVec2(5, -5), true,
                                               "Freq: %.2f Hz, Voice-coil "
                                               "impedance phase: %.1f deg",
                                               f, phzvc);
                        }

                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::Separator();
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Sound Pressure Level")) {
                    if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
                        ImPlot::SetupAxes("Frequency [Hz]", "Sound Pressure Level [dB]", ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight, ImPlotAxisFlags_NoSideSwitch | ImPlotAxisFlags_NoHighlight);
                        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                        ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(), ImPlotCond_Always);
                        const auto [min, max] = std::ranges::minmax(spl);
                        const double d = (max - min) * .1;
                        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, min - d, max + d);
                        ImPlot::PlotLine("Driver SPL", freq.data(), spld.data(), nfreq);
                        ImPlot::PlotLine("Port SPL", freq.data(), splp.data(), nfreq);
                        ImPlot::PlotLine("Overall SPL", freq.data(), spl.data(), nfreq);

                        static bool crosshairs_spld = false;
                        static bool crosshairs_splp = false;
                        static bool crosshairs_spl = true;

                        if ((crosshairs_spld || crosshairs_splp || crosshairs_spl) &&
                            ImPlot::IsPlotHovered() && show_сrosshairs) {
                            double f = ImPlot::GetPlotMousePos().x;
                            ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                            double spld, splp, spl;
                            calc_vented_box(f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &spld, &splp, &spl);
                            if (crosshairs_spld) {
                                ImPlot::DragLineY(id1, &spld, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                                ImPlot::Annotation(f, spld, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Driver SPL: %.1f dB", f, spld);
                            } else if (crosshairs_splp) {
                                ImPlot::DragLineY(id1, &splp, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                                ImPlot::Annotation(f, splp, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Port SPL: %.1f dB", f, splp);
                            } else if (crosshairs_spl) {
                                ImPlot::DragLineY(id1, &spl, ImVec4(.9f, .15f, .15f, .5f), 1.f, ImPlotDragToolFlags_NoInputs);
                                ImPlot::Annotation(f, spl, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5), true, "Freq: %.2f Hz, Overall SPL: %.1f dB", f, spl);
                            }
                        }

                        if (ImGui::BeginPopupContextItem()) {
                            ImGui::MenuItem("Driver database", NULL, &show_driver);
                            ImGui::MenuItem("Enclosure design", NULL, &show_design);
                            ImGui::MenuItem("Options", NULL, &show_options);
                            ImGui::MenuItem("Crosshairs", NULL, &show_сrosshairs);
                            ImGui::Separator();

                            if (!show_сrosshairs) {
                                ImGui::BeginDisabled();
                            }

                            if (ImGui::BeginMenu("Crosshairs to...")) {
                                if (ImGui::MenuItem("Driver SPL", NULL, &crosshairs_spld)) {
                                    if (crosshairs_spld) {
                                        crosshairs_splp = false;
                                        crosshairs_spl = false;
                                    } else {
                                        crosshairs_spld = true;
                                    }
                                }
                                if (ImGui::MenuItem("Port SPL", NULL, &crosshairs_splp)) {
                                    if (crosshairs_splp) {
                                        crosshairs_spld = false;
                                        crosshairs_spl = false;
                                    } else {
                                        crosshairs_splp = true;
                                    }
                                }
                                if (ImGui::MenuItem("Overall SPL", NULL, &crosshairs_spl)) {
                                    if (crosshairs_spl) {
                                        crosshairs_spld = false;
                                        crosshairs_splp = false;
                                    } else {
                                        crosshairs_spl = true;
                                    }
                                }
                                ImGui::EndMenu();
                            }
                            if (!show_сrosshairs) {
                                ImGui::EndDisabled();
                            }
                            ImGui::EndPopup();
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            ImPlot::PopStyleColor();
            ImGui::End();
        }

        if (show_options) {
            ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, FLT_MAX));
            if (ImGui::Begin("Options", &show_options, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse)) {
                static const double rg_min = .0;
                ImGui::DragScalar("Resistance of source", ImGuiDataType_Double, &rg, .01f, &rg_min, nullptr, "Rg: %.2f ohms", ImGuiSliderFlags_AlwaysClamp);

                static const double c_min = 300.;
                ImGui::DragScalar("Velocity of sound in air", ImGuiDataType_Double, &c, .1f, &c_min, nullptr, "c: %.1f m/sec", ImGuiSliderFlags_AlwaysClamp);

                static const double rh0_min = 1.;
                ImGui::DragScalar("Density of air", ImGuiDataType_Double, &rh0, .01f, &rh0_min, nullptr, "Rh0: %.2f kg/m3", ImGuiSliderFlags_AlwaysClamp);
            }
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}