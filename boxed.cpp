#include <complex>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl2.h"
#include "imgui.h"
#include "implot.h"
#include "nlohmann/json.hpp"

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

template <class T, int P> static T round(T a) {
  static_assert(std::is_floating_point<T>::value,
                "Round<T>: T must be floating point");
  const T shift = std::pow(static_cast<T>(10.0), P);
  return std::round(a * shift) / shift;
}

template <class T, int ULP>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y) {
  return std::fabs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ULP ||
         std::fabs(x - y) < std::numeric_limits<T>::min();
}

static double fmin(const std::function<double(double)> &f, double &a,
                   double b) {
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
fzero(const std::function<double(double)> &f, double &b, double c,
      const double tolerance = std::numeric_limits<double>::epsilon()) {
  short ic = 0;

  double z = c;
  double t = b;

  if (z == t) {
    return 4;
  }

  double fb = f(t);
  int count = 1;

  if (std::fabs(fb) <= tolerance) { // Zero at b
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

    if (std::fabs(fc) <= tolerance) { // Zero at c
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
    if (fabs(fc) < fabs(fb)) { // Interchange if necessary
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
      if (p <= tol * fabs(q)) { // Test for too small a change
        b += tol * sign(cmb);
      } else { // Root between b and (b + c)/2
        if (p < cmb * q) {
          b += p / q; // Use secant rule
        } else {
          b = 0.5 * (c + b);
        }
      }
    } // End else !((ic >= 4) && (8 * acmb >= acbs))

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

  const std::filesystem::path path("./drivers");

  static std::vector<std::filesystem::directory_entry> filenames;
  static std::vector<std::filesystem::directory_entry>::iterator it_filenames;
  const auto load_filenames = [&path]() {
    filenames.clear();
    std::copy_if(std::filesystem::directory_iterator{path},
                 std::filesystem::directory_iterator(),
                 back_inserter(filenames), [](const auto &entry) {
                   return std::filesystem::is_regular_file(entry) &&
                          entry.path().extension() == ".json";
                 });
    it_filenames = filenames.begin();
  };

  load_filenames();

  if (filenames.empty()) {
    return 1;
  }

  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    return 1;
  }

  GLFWwindow *window = glfwCreateWindow(1280, 720, "Boxed", NULL, NULL);
  if (window == NULL) {
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
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
    bool operator==(const driver_model &) const = default;
  };

  struct vented_box {
    double g;
    double phg;
    double zvc;
    double phzvc;
    double gd;
    double x;
    double v;
    double spld;
    double splp;
    double spl;
  };

  constexpr int id0 = 14666;
  constexpr int id1 = 14666;
  constexpr int id2 = 14668;

  static double rg = 0.01; /* Output resistance of source or amplifier */
  static double ql = 7.;
  static double vb;         /* Box Volume */
  static double fb;         /* Box Tuning Frequency */
  static double sp = .0001; /* Port Area */
  static const double sp_min = .0001;
  static double eg = .0;     /* Open-circuit output voltage of source */
  static double k = 0.732;   /* End Correction factor */
  static double rh0 = 1.184; /* Dencity of air */
  static double c = 346.1;   /* Velocity of sound in air */

  static const double f64_zero = 0.;

  constexpr int nfreq = 400;
  constexpr double low = 10.;
  constexpr double high = 1000.;
  static std::array<double, nfreq> freq;
  static const double gap =
      std::exp((std::log(high) - std::log(low)) / (nfreq));
  freq[0] = low;
  for (int i = 1; i < nfreq; ++i) {
    freq[i] = freq[i - 1] * gap;
  }

  static std::map<std::string, std::vector<driver_model>> vendors;
  const auto try_load_files = [&path]() {
    const std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    while (it_filenames != filenames.end()) {

      const std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count() > 10) {
        break;
      }

      nlohmann::json json;
      try {
        std::ifstream(*it_filenames++) >> json;
      } catch (nlohmann::json::parse_error) {
        continue;
      }

      std::string vendor{};
      driver_model driver{};
      for (const auto &[key, value] : json.items()) {
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

      if(driver.cms == .0 && driver.fs != .0 && driver.mms != .0) {
        driver.cms = std::pow( 1. / (driver.fs * 2. * std::numbers::pi), 2. ) / driver.mms;
      }

      if (driver.rms == .0 && driver.qms != .0) {
        driver.rms = std::sqrt(driver.mms / driver.cms) / driver.qms;
      }

      if (driver.sd == .0 || driver.cms == .0 || driver.mms == .0 ||
          driver.bl == .0 || driver.re == .0) {
        continue;
      }

      if (!vendor.empty() && !driver.model.empty()) {
        const auto drivers = vendors.find(vendor);
        if (drivers != vendors.end()) {
          drivers->second.emplace_back(driver);
        } else {
          vendors.insert({vendor, {driver}});
        }
      }
    }
  };

  while (!glfwWindowShouldClose(window)) {

    try_load_files();

    glfwPollEvents();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    static bool show_driver = true;
    static bool show_design = true;
    static bool show_options = true;
    static bool show_сrosshairs = true;

    static bool fit_g = true;
    static bool fit_phg = true;
    static bool fit_zvc = true;
    static bool fit_phzvc = true;
    static bool fit_gd = true;
    static bool fit_x = true;
    static bool fit_v = true;
    static bool fit_spl = true;

    static driver_model driver{};

    static int index_vendor = 0;
    static int index_model = 0;
    static int index_vendor_prev = -1;
    static int index_model_prev = -1;

    if (show_driver) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, FLT_MAX));
      if (ImGui::Begin("Driver database", &show_driver,
                       ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushItemWidth(220);
        ImGui::BeginGroup();
        ImGui::Combo(
            "Vendor", &index_vendor,
            [](void *map, int idx, const char **out_text) {
              auto it =
                  static_cast<
                      std::map<std::string, std::vector<driver_model>> *>(map)
                      ->cbegin();
              std::advance(it, idx);
              *out_text = it->first.data();
              return true;
            },
            static_cast<void *>(&vendors), static_cast<int>(vendors.size()),
            25);

        if (index_vendor_prev != index_vendor) {
          index_model = 0;
        }

        auto it_vendor = vendors.begin();
        std::advance(it_vendor, index_vendor);

        ImGui::Combo(
            "Model", &index_model,
            [](void *vector, int idx, const char **out_text) {
              auto it =
                  static_cast<std::vector<driver_model> *>(vector)->cbegin();
              std::advance(it, idx);
              *out_text = it->model.data();
              return true;
            },
            static_cast<void *>(&it_vendor->second),
            static_cast<int>(it_vendor->second.size()), 25);

        auto it_driver = it_vendor->second.begin();
        std::advance(it_driver, index_model);

        if (index_vendor_prev != index_vendor ||
            index_model_prev != index_model) {
          driver = *it_driver;
          fit_g = true;
          fit_phg = true;
          fit_zvc = true;
          fit_phzvc = true;
          fit_gd = true;
          fit_x = true;
          fit_v = true;
          fit_spl = true;
        }

        ImGui::EndGroup();

        ImGui::SameLine();

        ImGui::BeginGroup();
        if (ImGui::Button("Reload", ImVec2(-1.f, 0.f))) {
          vendors.clear();
          load_filenames();
          try_load_files();

          index_vendor_prev = -1;
          index_model_prev = -1;
          index_vendor = 0;
          index_model = 0;

          it_vendor = vendors.begin();
          it_driver = it_vendor->second.begin();
          driver = *it_driver;

          fit_g = true;
          fit_phg = true;
          fit_zvc = true;
          fit_phzvc = true;
          fit_gd = true;
          fit_x = true;
          fit_v = true;
          fit_spl = true;
        }

        const bool driver_modified = (driver != *it_driver);

        if (!driver_modified) {
          ImGui::BeginDisabled();
        }
        const ImVec2 size = ImVec2(50.f, 0.f);
        if (ImGui::Button("Save", size)) {
          *it_driver = driver;

          nlohmann::json json;
          json["vendor"] = it_vendor->first;
          json["model"] = it_driver->model;
          json["mms"] = round<double, 4>(it_driver->mms);
          json["cms"] = round<double, 7>(it_driver->cms);
          json["re"] = round<double, 3>(it_driver->re);
          json["bl"] = round<double, 5>(it_driver->bl);
          if (it_driver->rms > .0) {
            json["rms"] = round<double, 5>(it_driver->rms);
          }
          json["sd"] = round<double, 5>(it_driver->sd);
          json["le"] = round<double, 6>(it_driver->le);
          if (it_driver->fs > .0) {
            json["fs"] = round<double, 3>(it_driver->fs);
          }
          if (it_driver->qes > .0) {
            json["qes"] = round<double, 3>(it_driver->qes);
          }
          if (it_driver->qms > .0) {
            json["qms"] = round<double, 3>(it_driver->qms);
          }
          if (it_driver->qts > .0) {
            json["qts"] = round<double, 3>(it_driver->qts);
          }
          json["vas"] = round<double, 5>(it_driver->vas);
          if (it_driver->xmax > .0) {
            json["xmax"] = round<double, 4>(it_driver->xmax);
          }
          if (it_driver->xmech > .0) {
            json["xmech"] = round<double, 4>(it_driver->xmech);
          }
          if (it_driver->pe > .0) {
            json["pe"] = round<double, 1>(it_driver->pe);
          }
          if (it_driver->depth > .0) {
            json["depth"] = round<double, 1>(it_driver->depth);
          }
          if (it_driver->mdepth > .0) {
            json["mdepth"] = round<double, 1>(it_driver->mdepth);
          }
          if (it_driver->mdia > .0) {
            json["mdia"] = round<double, 1>(it_driver->mdia);
          }
          if (it_driver->vcd > .0) {
            json["vcd"] = round<double, 1>(it_driver->vcd);
          }
          std::ofstream o(it_vendor->first + " " + it_driver->model + ".json");
          o << std::setw(4) << json << std::endl;
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset", size)) {
          driver = *it_driver;
        }

        if (!driver_modified) {
          ImGui::EndDisabled();
        }

        ImGui::EndGroup();

        ImGui::PopItemWidth();

        ImGui::PushItemWidth(135);

        if (ImGui::CollapsingHeader("Fundamental parameters")) {
          static const double sd_min = .00001;
          ImGui::DragScalar("Surface Area Of Cone", ImGuiDataType_Double,
                            &driver.sd, .00001f, &sd_min, nullptr,
                            "Sd: %.5f m2", ImGuiSliderFlags_AlwaysClamp);

          static const double mms_min = .0001;
          ImGui::DragScalar("Diaphragm Mass Including Air-Load",
                            ImGuiDataType_Double, &driver.mms, .0001f, &mms_min,
                            nullptr, "Mms: %.4f kg",
                            ImGuiSliderFlags_AlwaysClamp);

          static const double cms_min = .0000001;
          ImGui::DragScalar("Mechanical Compliance Of Suspension",
                            ImGuiDataType_Double, &driver.cms, .0000001f,
                            &cms_min, nullptr, "Cms: %.7f m/N",
                            ImGuiSliderFlags_AlwaysClamp);

          static const double rms_min = .00001;
          ImGui::DragScalar("Mechanical Resistance", ImGuiDataType_Double,
                            &driver.rms, .00001f, &rms_min, nullptr,
                            "Rms: %.5f kg/s", ImGuiSliderFlags_AlwaysClamp);

          static const double le_min = .0;
          ImGui::DragScalar("Voice coil inductance", ImGuiDataType_Double,
                            &driver.le, .000001f, &le_min, nullptr,
                            "Le: %.6f H", ImGuiSliderFlags_AlwaysClamp);

          static const double re_min = .001;
          ImGui::DragScalar("DC resistance of the voice coil",
                            ImGuiDataType_Double, &driver.re, .001f, &re_min,
                            nullptr, "Re: %.3f ohms",
                            ImGuiSliderFlags_AlwaysClamp);

          static const double bl_min = .1;
          ImGui::DragScalar("BL Product", ImGuiDataType_Double, &driver.bl,
                            .00001f, &bl_min, nullptr, "BL: %.5f Tm",
                            ImGuiSliderFlags_AlwaysClamp);
        }

        if (ImGui::CollapsingHeader("Small signal parameters")) {
          static const double fs_min = 10.;
          ImGui::DragScalar("Resonant Frequency", ImGuiDataType_Double,
                            &driver.fs, .001f, &fs_min, nullptr, "Fs: %.3f Hz",
                            ImGuiSliderFlags_AlwaysClamp);

          double fs =
              1. / (2. * std::numbers::pi * std::sqrt(driver.cms * driver.mms));
          ImGui::BeginDisabled();
          ImGui::DragScalar("Resonant Frequency'", ImGuiDataType_Double, &fs,
                            .0f, nullptr, nullptr, "Fs': %.3f Hz");
          ImGui::EndDisabled();

          static const double qms_min = .001;
          ImGui::DragScalar("Mechanical Q", ImGuiDataType_Double, &driver.qms,
                            .001f, &qms_min, nullptr, "Qms: %.3f",
                            ImGuiSliderFlags_AlwaysClamp);

          double qms = std::sqrt(driver.mms / driver.cms) / driver.rms;
          ImGui::BeginDisabled();
          ImGui::DragScalar("Mechanical Q'", ImGuiDataType_Double, &qms, .0f,
                            nullptr, nullptr, "Qms': %.3f");
          ImGui::EndDisabled();

          static const double qes_min = .001;
          ImGui::DragScalar("Electrical Q", ImGuiDataType_Double, &driver.qes,
                            .001f, &qes_min, nullptr, "Qes: %.3f",
                            ImGuiSliderFlags_AlwaysClamp);

          double qes = std::sqrt(driver.mms / driver.cms) * driver.re /
                       std::pow(driver.bl, 2);
          ImGui::BeginDisabled();
          ImGui::DragScalar("Electrical Q'", ImGuiDataType_Double, &qes, .0f,
                            nullptr, nullptr, "Qes': %.3f");
          ImGui::EndDisabled();

          static const double qts_min = .001;
          ImGui::DragScalar("Total Q", ImGuiDataType_Double, &driver.qts, .001f,
                            &qts_min, nullptr, "Qts: %.3f",
                            ImGuiSliderFlags_AlwaysClamp);

          double qts = qms * qes / (qms + qes);
          ImGui::BeginDisabled();
          ImGui::DragScalar("Total Q'", ImGuiDataType_Double, &qts, .0f,
                            nullptr, nullptr, "Qts': %.3f");
          ImGui::EndDisabled();

          static const double vas_min = .00001;
          ImGui::DragScalar("Equivalent Compliance Volume",
                            ImGuiDataType_Double, &driver.vas, .00001f,
                            &vas_min, nullptr, "Vas: %.5f m3",
                            ImGuiSliderFlags_AlwaysClamp);

          double vas =
              rh0 * std::pow(c, 2.) * std::pow(driver.sd, 2.) * driver.cms;
          ImGui::BeginDisabled();
          ImGui::DragScalar("Equivalent Compliance Volume'",
                            ImGuiDataType_Double, &vas, .0f, nullptr, nullptr,
                            "Vas': %.5f m3");
          ImGui::EndDisabled();
        }

        if (ImGui::CollapsingHeader("Large signal parameters")) {
          static const double xmax_min = .0;
          ImGui::DragScalar("Maximum Linear Excursion", ImGuiDataType_Double,
                            &driver.xmax, 0.0001f, &xmax_min, nullptr,
                            "Xmax: %.4f m", ImGuiSliderFlags_AlwaysClamp);

          static const double xmech_min = .0;
          ImGui::DragScalar("Maximum Mechanical Excursion",
                            ImGuiDataType_Double, &driver.xmech, 0.0001f,
                            &xmech_min, nullptr, "Xmech: %.4f m",
                            ImGuiSliderFlags_AlwaysClamp);

          static const double pe_min = 1.;
          ImGui::DragScalar("Thermal Power Handling", ImGuiDataType_Double,
                            &driver.pe, 0.1f, &pe_min, nullptr, "Pe: %.1f W",
                            ImGuiSliderFlags_AlwaysClamp);
        }

        if (ImGui::CollapsingHeader("Dimensions")) {
          ImGui::DragScalar("Depth", ImGuiDataType_Double, &driver.depth, 0.1f,
                            &f64_zero, nullptr, "%.1f mm",
                            ImGuiSliderFlags_AlwaysClamp);

          ImGui::DragScalar("Magnet Depth", ImGuiDataType_Double,
                            &driver.mdepth, 0.1f, &f64_zero, nullptr, "%.1f mm",
                            ImGuiSliderFlags_AlwaysClamp);

          ImGui::DragScalar("Magnet Diameter", ImGuiDataType_Double,
                            &driver.mdia, 0.1f, &f64_zero, nullptr, "%.1f mm",
                            ImGuiSliderFlags_AlwaysClamp);

          ImGui::DragScalar("Voice Coil Diameter", ImGuiDataType_Double,
                            &driver.vcd, 0.1f, &f64_zero, nullptr, "%.1f mm",
                            ImGuiSliderFlags_AlwaysClamp);

          static double vol = .0;
          /* magnet volume */
          const double mv =
              std::numbers::pi * std::pow(driver.mdia / 2., 2.) * driver.mdepth;
          /* surface area radius */
          const double sr = driver.sd / 2.;
          /* voice-coil radius */
          const double vcr = driver.vcd / 2.;
          /* cone height */
          double ch = driver.depth - driver.mdepth - driver.xmax;
          if (ch < .0)
            ch = .0;
          /* cone volume */
          const double cv = std::numbers::pi * ch *
                            (std::pow(vcr, 2.) + std::pow(sr, 2.) + vcr * sr) /
                            3;
          /* total volume */
          vol = (mv + cv) / 1e+6;
          ImGui::BeginDisabled();
          ImGui::DragScalar("Driver volume", ImGuiDataType_Double, &vol, .0f,
                            nullptr, nullptr, "%.3f L");
          ImGui::EndDisabled();
        }
        ImGui::PopItemWidth();
        ImGui::End();
      }
    }

    const double sd2 = std::pow(driver.sd, 2.);
    const double bl2 = std::pow(driver.bl, 2.);

    const double eg_max = std::sqrt(driver.pe * driver.re);

    if (index_vendor_prev != index_vendor || index_model_prev != index_model) {
      eg = (eg_max > .0) ? eg_max : 1.;
      const double fs = (driver.fs > .0)
                            ? driver.fs
                            : 1. / (2. * std::numbers::pi *
                                    std::sqrt(driver.cms * driver.mms));
      const double vas = (driver.vas > .0)
                             ? driver.vas
                             : rh0 * std::pow(c, 2.) * sd2 * driver.cms;
      const double qes =
          (driver.qes > .0)
              ? driver.qes
              : std::sqrt(driver.mms / driver.cms) * driver.re / bl2;
      const double qms = (driver.qms > .0)
                             ? driver.qms
                             : std::sqrt(driver.mms / driver.cms) / driver.rms;
      const double qts =
          (driver.qts > .0) ? driver.qts : qms * qes / (qms + qes);
      vb = vas * std::pow(qts / 0.4, 3.) / 1.1;
      fb = fs * 0.42 / qts;
    }

    const auto calc_lpa = [&]() {
      return std::pow(c, 2.) * sp /
             (std::pow(2. * std::numbers::pi, 2.) * std::pow(fb, 2.) * vb);
    };

    if (show_design) {
      ImGui::SetNextWindowSizeConstraints(ImVec2(400, 0), ImVec2(400, FLT_MAX));
      if (ImGui::Begin("Enclosure design", &show_design,
                       ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoCollapse)) {

        ImGui::PushItemWidth(180);

        if (ImGui::CollapsingHeader("Box", ImGuiTreeNodeFlags_SpanFullWidth)) {
          static const double vbl_min = 1.;
          double vbl = 1000. * vb;
          ImGui::DragScalar("Volume", ImGuiDataType_Double, &vbl, .01f,
                            &vbl_min, nullptr, "Vb %.2f L",
                            ImGuiSliderFlags_AlwaysClamp);
          vb = vbl / 1000.;

          static const double fb_min = 10.;
          ImGui::DragScalar("Tuning Frequency", ImGuiDataType_Double, &fb,
                            0.01f, &fb_min, nullptr, "Fb %.2f Hz",
                            ImGuiSliderFlags_AlwaysClamp);

          static const double ql_max = 30.;
          static const double ql_min = 3.;
          ImGui::SliderScalar("Leakage Q", ImGuiDataType_Double, &ql, &ql_min,
                              &ql_max, "Ql %.f", ImGuiSliderFlags_AlwaysClamp);
        }

        if (ImGui::CollapsingHeader("Port")) {
          ImGui::DragScalar("Area", ImGuiDataType_Double, &sp, 0.00001f,
                            &sp_min, nullptr, "Sp %.5f m2",
                            ImGuiSliderFlags_AlwaysClamp);

          double dp = 2. * std::sqrt(sp / std::numbers::pi);
          ImGui::BeginDisabled();
          ImGui::DragScalar("Diameter", ImGuiDataType_Double, &dp, .0f, nullptr,
                            nullptr, "Dp %.4f m");

          static double hp = .01;
          double wp = sp / hp;
          if (wp < 0.01) {
            wp = 0.01;
          }
          ImGui::DragScalar("Width", ImGuiDataType_Double, &wp, .0f, nullptr,
                            nullptr, "Wp %.3f m");
          ImGui::EndDisabled();

          const double hp_max = std::sqrt(sp);
          if (hp > hp_max) {
            hp = hp_max;
          }
          static const double hp_min = .01;
          ImGui::SliderScalar("Height", ImGuiDataType_Double, &hp, &hp_min,
                              &hp_max, "Hp %.3f m",
                              ImGuiSliderFlags_AlwaysClamp);

          static const double k_min = 0.;
          ImGui::DragScalar("End correction", ImGuiDataType_Double, &k, .0001f,
                            &k_min, nullptr, "k %.3f");

          double lpa = calc_lpa();
          double lpm = lpa - k * dp;
          if (lpm < .0) {
            lpm = .0;
          }
          ImGui::BeginDisabled();
          ImGui::DragScalar("Length", ImGuiDataType_Double, &lpm, .0f, nullptr,
                            nullptr, "Lp %.3f m");
          ImGui::DragScalar("Length acoustical", ImGuiDataType_Double, &lpa,
                            .0f, nullptr, nullptr, "Lpa %.3f m");
          ImGui::EndDisabled();
        }
        ImGui::PopItemWidth();
        ImGui::End();
      }
    }

    if (eg_max > .0 && eg > eg_max) {
      eg = eg_max;
    }

    /* Cab - Acoustic compliance of air in enclosure */
    const double cab = vb / (rh0 * c * c);

    /* Cas - acoustic compliance of driver suspension */
    const double cas =
        (driver.vas > .0) ? driver.vas / (rh0 * c * c) : sd2 * driver.cms;

    /* Wb */
    const double wb = 2. * std::numbers::pi * fb;
    /* Cmes - electrical capacitance due to driver mass */
    const double cmes = driver.mms / bl2;
    /* Lces - electrical inductance due to driver compliance */
    const double lces = driver.cms * bl2;
    /* Ras - acoustic resistance of driver suspension losses */
    const double ras = driver.rms / sd2;
    /* Res - electrical resistance due to driver suspension losses */
    const double res = bl2 / driver.rms;
    /* Lceb */
    const double lceb = bl2 * cab / sd2;
    /* Ral - acoustic resistance of enclosure lossescaused by leakage */
    const double ral = ql / (wb * cab);
    /* Rel - acoustic resistance of enclosure lossescaused by leakage */
    const double rel = bl2 / (sd2 * ral);
    /* Mas - acoustic mass of driver diaphragm assembly including air load */
    const double mas = driver.mms / sd2;
    /* Pg - acoustic driving pressure */
    const double pg = eg * driver.bl / ((rg + driver.re) * driver.sd);

    const double eg_spl = std::sqrt(driver.re);
    const double pg_spl = eg_spl * driver.bl / ((rg + driver.re) * driver.sd);

    /* Rat - acoustic resistance of total driver-circuit losses */
    const double rat = ras + bl2 / (sd2 * (rg + driver.re));

    std::function<vented_box(const double)> calc_vented_box;

    calc_vented_box = [&](const double f) {
      const auto phase = [](const std::complex<double> &arg) {
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

      const auto parallel2 = [](const auto z0, const auto z1) {
        return (z0 * z1) / (z0 + z1);
      };

      const auto parallel3 = [](const auto z0, const auto z1, const auto z2) {
        return (z0 * z1 * z2) / (z0 * z1 + z1 * z2 + z2 * z0);
      };

      /* Map - acoustic mass of port including air load */
      const double map = calc_lpa() * rh0 / sp;
      /* Cmep */
      const double cmep = cab * map / lceb;

      const std::complex s(0., f * 2. * std::numbers::pi);
      const std::complex zaa = s * map;
      const std::complex zas = rat + s * mas + 1. / (s * cas);
      const std::complex zab = 1. / (s * cab);

      const std::complex z0 = parallel2(zab, ral);

      /* Up - volume velocity of port */
      const std::complex up = pg * z0 / (zaa * z0 + zas * (zaa + z0));

      const std::complex up_spl = pg_spl * z0 / (zaa * z0 + zas * (zaa + z0));

      const std::complex z1 = zas + parallel3(zaa, zab, ral);
      const std::complex z2 = parallel2(zaa, ral);

      /* U0 - total volume velocity leaving enclosure boundaries */
      const std::complex u0 = pg * z2 / (zab * z2 + zas * (zab + z2));

      vented_box enclosure;

      /* volume velocity of driver diaphragm */
      const std::complex ud = pg / z1;
      const std::complex ud_spl = pg_spl / z1;
      /* Diaphragm dsiplacement */
      enclosure.x = std::sqrt(2.) * std::abs(ud / (driver.sd * s));
      /* Response */
      const std::complex g = s * mas * u0 / pg;
      enclosure.g = 20. * std::log10(std::abs(g));
      /* Response phase */
      const double phg = phase(g);
      enclosure.phg = phg;

      const double f_prev = f / gap;
      const std::complex s_prev(0., f_prev * 2. * std::numbers::pi);
      const std::complex zas_prev = rat + s_prev * mas + 1. / (s_prev * cas);
      const std::complex zaa_prev = s_prev * map;
      const std::complex zab_prev = 1. / (s_prev * cab);

      const std::complex z3 = parallel2(zaa_prev, ral);

      const std::complex u0_prev =
          pg * z3 / (zab_prev * z3 + zas_prev * (zab_prev + z3));
      const std::complex g_prev = s_prev * mas * u0_prev / pg;
      const double phg_prev = phase(g_prev);
      /* Group delay */
      if ((phg > .0 && phg_prev > .0) || (phg < .0 && phg_prev < .0)) {
        enclosure.gd = (-1. / 360.) * ((phg - phg_prev) / (f - f_prev));
      } else {
        const double gd_prev = calc_vented_box(f_prev).gd;
        const double gd_next = calc_vented_box(f * gap).gd;
        enclosure.gd = .5 * (gd_prev + gd_next);
      }
      /* Port air velocity */
      enclosure.v = std::sqrt(2.) * std::abs(up) / sp;
      /* Driver sound pressure level */
      enclosure.spld = 79.6 + 20. * std::log10(std::abs(ud_spl * rh0 * s));
      /* Port sound pressure level */
      enclosure.splp = 79.6 + 20. * std::log10(std::abs(up_spl * rh0 * s));
      /* Overall sound pressure level */
      enclosure.spl =
          79.6 + 20. * std::log10(std::abs((up_spl - ud_spl) * s * rh0));

      const std::complex ze = rel + s * lceb + 1. / (s * cmep);
      const std::complex zs = parallel3(res, s * lces, 1. / (s * cmes));
      const std::complex zm = parallel2(ze, zs);
      /* Voice-coil impedance */
      const std::complex zvc = driver.re + s * driver.le + zm;
      enclosure.zvc = std::abs(zvc);
      /* Voice-coil impedance phase */
      enclosure.phzvc = phase(zvc);

      return enclosure;
    };

    std::array<double, nfreq> g, phg, zvc, phzvc, gd, x, v, spld, splp, spl;

    if (index_vendor_prev != index_vendor || index_model_prev != index_model) {
      const double sp_prev = sp;
      const auto f = [&](const double arg) {
        sp = arg;
        double a = freq.front();
        return fmin([&](const double arg) { return calc_vented_box(arg).v; }, a,
                    freq.back()) -
               9.;
      };
      double b = sp_min;
      const int r = fzero(f, b, 2 * driver.sd);
      if (r == 5) {
        sp = sp_prev;
      }
    }

    index_vendor_prev = index_vendor;
    index_model_prev = index_model;

    for (int i = 0; i < freq.size(); ++i) {
      const vented_box enclosure = calc_vented_box(freq[i]);
      g[i] = enclosure.g;
      phg[i] = enclosure.phg;
      zvc[i] = enclosure.zvc;
      phzvc[i] = enclosure.phzvc;
      gd[i] = enclosure.gd;
      x[i] = enclosure.x;
      v[i] = enclosure.v;
      spld[i] = enclosure.spld;
      splp[i] = enclosure.splp;
      spl[i] = enclosure.spl;
    }

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    if (ImGui::Begin("Plot", NULL,
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                         ImGuiWindowFlags_NoDecoration |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoSavedSettings)) {
      const ImPlotFlags flagsPlot =
          ImPlotFlags_NoTitle | ImPlotFlags_AntiAliased | ImPlotFlags_NoLegend |
          ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect |
          ImPlotFlags_NoMouseText;
      ImPlot::PushStyleColor(ImPlotCol_FrameBg, IM_COL32_BLACK_TRANS);

      if (ImGui::BeginTabBar("Plots")) {
        if (ImGui::BeginTabItem("Response")) {
          if (ImPlot::BeginPlot("##plot", ImVec2(-1, -1), flagsPlot)) {
            ImPlot::SetupAxes("Frequency [Hz]", "Response [dB]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);
            const auto [min, max] = std::minmax_element(g.begin(), g.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_g ? ImPlotCond_Always
                                          : ImPlotCond_Once);
            fit_g = false;
            ImPlot::PlotLine("##line", freq.data(), g.data(), nfreq);
            static bool show_f3 = true;
            if (show_f3) {
              double a = freq.front();
              const int r = fzero(
                  [&](const double v) { return calc_vented_box(v).g + 3.; }, a,
                  freq.back());
              if (r != 5) {
                ImPlot::DragLineX(id0, &a, ImVec4(.15f, .8f, .15f, .5f), 1.f,
                                  ImPlotDragToolFlags_NoInputs);
                const ImPlotRect rect = ImPlot::GetPlotLimits();
                ImPlot::Annotation(a, rect.Min().y, ImVec4(.15f, .15f, .15f, 1),
                                   ImVec2(5, -5), true,
                                   "Freq: %.2f Hz, Response: %.2f dB", a,
                                   calc_vented_box(a).g);
              }
            }
            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id1, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id2, &enclosure.g, ImVec4(.9f, .15f, .15f, .5f),
                                1.f, ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(
                  f, enclosure.g, ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                  true, "Freq: %.2f Hz, Response: %.2f dB", f, enclosure.g);
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
            ImPlot::SetupAxes("Frequency [Hz]", "Response phase [deg]",
                              ImPlotAxisFlags_LogScale);
            const auto [min, max] = std::minmax_element(phg.begin(), phg.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_phg ? ImPlotCond_Always
                                            : ImPlotCond_Once);
            fit_phg = false;
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);
            ImPlot::PlotLine("##line", freq.data(), l0.data(),
                             static_cast<int>(l0.size()));
            ImPlot::PlotLine("##line", freq.data() + l0.size(), l1.data(),
                             static_cast<int>(l1.size()));
            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id1, &enclosure.phg,
                                ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, enclosure.phg, ImVec4(.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Response phase: %.2f deg", f,
                                 enclosure.phg);
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
            ImPlot::SetupAxes("Frequency [Hz]", "Group delay [sec]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);
            const auto [min, max] = std::minmax_element(gd.begin(), gd.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_gd ? ImPlotCond_Always
                                           : ImPlotCond_Once);
            fit_gd = false;
            ImPlot::PlotLine("##line", freq.data(), gd.data(), nfreq);
            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id1, &enclosure.gd,
                                ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, enclosure.gd, ImVec4(.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Group delay: %.4f sec", f,
                                 enclosure.gd);
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
            ImPlot::SetupAxes("Frequency [Hz]", "Cone excursion [m]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);
            const auto [min, max] = std::minmax_element(x.begin(), x.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_x ? ImPlotCond_Always
                                          : ImPlotCond_Once);
            fit_x = false;
            ImPlot::PlotLine("##line", freq.data(), x.data(), nfreq);
            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id1, &enclosure.x, ImVec4(.9f, .15f, .15f, .5f),
                                1.f, ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, enclosure.x, ImVec4(.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Cone excursion: %.4f m", f,
                                 enclosure.x);
            }
            static bool show_xmax = true;
            static bool show_xmech = true;
            if (driver.xmax > .0 && show_xmax) {
              ImPlot::DragLineY(id2, &driver.xmax, ImVec4(.15f, .9f, .15f, .5f),
                                1.f, ImPlotDragToolFlags_NoInputs);
              const ImPlotRect rect = ImPlot::GetPlotLimits();
              ImPlot::Annotation(rect.Max().x, driver.xmax,
                                 ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                                 true, "Xmax %.4f m", driver.xmax);
            }
            if (driver.xmech > .0 && show_xmech) {
              ImPlot::DragLineY(id2, &driver.xmech,
                                ImVec4(.15f, .9f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              const ImPlotRect rect = ImPlot::GetPlotLimits();
              ImPlot::Annotation(rect.Max().x, driver.xmech,
                                 ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                                 true, "Xmech %.4f m", driver.xmech);
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
            ImPlot::SetupAxes("Frequency [Hz]", "Port air velocity [m/sec]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);
            const auto [min, max] = std::minmax_element(v.begin(), v.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_v ? ImPlotCond_Always
                                          : ImPlotCond_Once);
            fit_v = false;
            ImPlot::PlotLine("##line", freq.data(), v.data(), nfreq);
            static bool show_max_v = true;
            if (show_max_v) {
              double a = freq.front();
              double max_v =
                  fmin([&](const double arg) { return calc_vented_box(arg).v; },
                       a, freq.back());
              ImPlot::DragLineX(id0, &a, ImVec4(.15f, .8f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              const ImPlotRect rect = ImPlot::GetPlotLimits();
              ImPlot::Annotation(a, rect.Min().y, ImVec4(.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Port air velocity: %.2f m/sec",
                                 a, max_v);
            }

            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id1, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id2, &enclosure.v, ImVec4(.9f, .15f, .15f, .5f),
                                1.f, ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, enclosure.v, ImVec4(.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Port air velocity: %.2f m/sec",
                                 f, enclosure.v);
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
            ImPlot::SetupAxes("Frequency [Hz]",
                              "Voice-coil impedance magnitude [ohms]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);

            const auto [min, max] = std::minmax_element(zvc.begin(), zvc.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_zvc ? ImPlotCond_Always
                                            : ImPlotCond_Once);
            fit_zvc = false;
            ImPlot::PlotLine("##line", freq.data(), zvc.data(), nfreq);

            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              ImPlot::DragLineY(id1, &enclosure.zvc,
                                ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, enclosure.zvc,
                                 ImVec4(0.15f, 0.15f, 0.15f, 1), ImVec2(5, -5),
                                 true,
                                 "Freq: %.2f Hz, Voice-coil "
                                 "impedance magnitude: %.2f ohms",
                                 f, enclosure.zvc);
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
            ImPlot::SetupAxes("Frequency [Hz]",
                              "Voice-coil impedance phase [deg]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);

            const auto [min, max] =
                std::minmax_element(phzvc.begin(), phzvc.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_phzvc ? ImPlotCond_Always
                                              : ImPlotCond_Once);
            fit_phzvc = false;
            ImPlot::PlotLine("##line", freq.data(), phzvc.data(), nfreq);

            if (show_сrosshairs && ImPlot::IsPlotHovered()) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box plot = calc_vented_box(f);
              ImPlot::DragLineY(id1, &plot.phzvc, ImVec4(.9f, .15f, .15f, .5f),
                                1.f, ImPlotDragToolFlags_NoInputs);
              ImPlot::Annotation(f, plot.phzvc, ImVec4(0.15f, .15f, .15f, 1),
                                 ImVec2(5, -5), true,
                                 "Freq: %.2f Hz, Voice-coil "
                                 "impedance phase: %.1f deg",
                                 f, plot.phzvc);
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

            ImPlot::SetupAxes("Frequency [Hz]", "Sound Pressure Level [dB]",
                              ImPlotAxisFlags_LogScale);
            ImPlot::SetupAxisLimits(ImAxis_X1, freq.front(), freq.back(),
                                    ImPlotCond_Always);

            const auto [min, max] = std::minmax_element(spl.begin(), spl.end());
            const double d = (*max - *min) * .1;
            ImPlot::SetupAxisLimits(ImAxis_Y1, *min - d, *max + d,
                                    fit_spl ? ImPlotCond_Always
                                            : ImPlotCond_Once);
            fit_spl = false;
            ImPlot::PlotLine("Driver SPL", freq.data(), spld.data(), nfreq);
            ImPlot::PlotLine("Port SPL", freq.data(), splp.data(), nfreq);
            ImPlot::PlotLine("Overall SPL", freq.data(), spl.data(), nfreq);

            static bool crosshairs_spld = false;
            static bool crosshairs_splp = false;
            static bool crosshairs_spl = true;

            if ((crosshairs_spld || crosshairs_splp || crosshairs_spl) &&
                ImPlot::IsPlotHovered() && show_сrosshairs) {
              double f = ImPlot::GetPlotMousePos().x;
              ImPlot::DragLineX(id0, &f, ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                ImPlotDragToolFlags_NoInputs);
              vented_box enclosure = calc_vented_box(f);
              if (crosshairs_spld) {
                ImPlot::DragLineY(id1, &enclosure.spld,
                                  ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                  ImPlotDragToolFlags_NoInputs);
                ImPlot::Annotation(f, enclosure.spld,
                                   ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                                   true, "Freq: %.2f Hz, Driver SPL: %.1f dB",
                                   f, enclosure.spld);
              } else if (crosshairs_splp) {
                ImPlot::DragLineY(id1, &enclosure.splp,
                                  ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                  ImPlotDragToolFlags_NoInputs);
                ImPlot::Annotation(f, enclosure.splp,
                                   ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                                   true, "Freq: %.2f Hz, Port SPL: %.1f dB", f,
                                   enclosure.splp);
              } else if (crosshairs_spl) {
                ImPlot::DragLineY(id1, &enclosure.spl,
                                  ImVec4(.9f, .15f, .15f, .5f), 1.f,
                                  ImPlotDragToolFlags_NoInputs);
                ImPlot::Annotation(f, enclosure.spl,
                                   ImVec4(.15f, .15f, .15f, 1), ImVec2(5, -5),
                                   true, "Freq: %.2f Hz, Overall SPL: %.1f dB",
                                   f, enclosure.spl);
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
      if (ImGui::Begin("Options", &show_options,
                       ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoCollapse)) {
        static const double eg_min = .1;
        if (eg_max > .0) {
          ImGui::SliderScalar("Voltage of source", ImGuiDataType_Double, &eg,
                              &eg_min, &eg_max, "eg: %.1f V",
                              ImGuiSliderFlags_AlwaysClamp);
        } else {
          ImGui::DragScalar("Voltage of source", ImGuiDataType_Double, &eg, .1f,
                            &eg_min, nullptr, "eg: %.1f V",
                            ImGuiSliderFlags_AlwaysClamp);
        }

        static const double rg_min = .0;
        ImGui::DragScalar("Resistance of source", ImGuiDataType_Double, &rg,
                          .01f, &rg_min, nullptr, "Rg: %.2f ohms",
                          ImGuiSliderFlags_AlwaysClamp);

        static const double c_min = 300.;
        ImGui::DragScalar("Velocity of sound in air", ImGuiDataType_Double, &c,
                          .1f, &c_min, nullptr, "c: %.1f m/sec",
                          ImGuiSliderFlags_AlwaysClamp);

        static const double rh0_min = 1.;
        ImGui::DragScalar("Density of air", ImGuiDataType_Double, &rh0, .01f,
                          &rh0_min, nullptr, "Rh0: %.2f kg/m3",
                          ImGuiSliderFlags_AlwaysClamp);
      }
      ImGui::End();
    }

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);

    const static ImVec4 clear_color = ImVec4(.45f, .55f, .6f, 1.f);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);

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