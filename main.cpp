#include "biharmonic_precompute.h"
#include "biharmonic_solve.h"
#include "arap_precompute.h"
#include "arap_single_iteration.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/read_triangle_mesh.h>
#include <Eigen/Core>
#include <iostream>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

// Helper function declarations
std::pair<glm::vec3, glm::vec3> unproject(const ImVec2& mousePos);
bool rayTriangleIntersect(const glm::vec3& orig, const glm::vec3& dir,
                         const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float &t);

// Global variables
Eigen::MatrixXd CV, CU; // Control vertices and their updated positions
bool placing_handles = true;
Eigen::MatrixXd V, U; // V: original mesh, U: deformed mesh
Eigen::MatrixXi F;
int sel = -1;
ImVec2 last_mouse;
igl::min_quad_with_fixed_data<double> biharmonic_data, arap_data;
Eigen::SparseMatrix<double> arap_K;
float vertexRadius = 0.008f;
bool cameraLocked = false;
glm::mat4 savedViewMatrix;

enum Method {
    BIHARMONIC = 0,
    ARAP = 1,
} method = BIHARMONIC;

// Helper: Convert a world-space position to screen coordinates.
glm::vec2 worldToScreen(const glm::vec3& worldPos) {
    glm::mat4 view = polyscope::view::getCameraViewMatrix();
    glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
    glm::vec4 clip = proj * view * glm::vec4(worldPos, 1.0f);
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    float x = (ndc.x + 1.0f) * 0.5f * ImGui::GetIO().DisplaySize.x;
    float y = (1.0f - ndc.y) * 0.5f * ImGui::GetIO().DisplaySize.y;
    return glm::vec2(x, y);
}

// Unproject a mouse position to obtain a ray (origin and normalized direction)
std::pair<glm::vec3, glm::vec3> unproject(const ImVec2& mousePos) {
    glm::mat4 projMatrix = polyscope::view::getCameraPerspectiveMatrix();
    glm::mat4 viewMatrix = polyscope::view::getCameraViewMatrix();
    glm::mat4 invProj = glm::inverse(projMatrix);
    glm::mat4 invView = glm::inverse(viewMatrix);

    float x = (2.0f * mousePos.x) / ImGui::GetIO().DisplaySize.x - 1.0f;
    float y = 1.0f - (2.0f * mousePos.y) / ImGui::GetIO().DisplaySize.y;

    glm::vec4 rayClip(x, y, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    glm::vec3 cameraPos = polyscope::view::getCameraWorldPosition();
    glm::vec3 rayDir = glm::normalize(glm::vec3(rayWorld));
    return {cameraPos, rayDir};
}

// Möller–Trumbore ray–triangle intersection algorithm
bool rayTriangleIntersect(
    const glm::vec3& orig,
    const glm::vec3& dir,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float &t)
{
    const float EPSILON = 1e-8f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false; // Parallel ray
    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;
    t = f * glm::dot(edge2, q);
    return (t > EPSILON);
}

bool find_closest_vertex(const ImVec2& mousePos, int& closest_vertex) {
    auto [rayOrigin, rayDir] = unproject(mousePos);
    
    // Find the first triangle that intersects with the ray
    float closest_t = std::numeric_limits<float>::max();
    int hit_triangle = -1;
    glm::vec3 intersection_point;
    
    // Check each triangle for intersection
    for (int i = 0; i < F.rows(); i++) {
        glm::vec3 v0(V(F(i, 0), 0), V(F(i, 0), 1), V(F(i, 0), 2));
        glm::vec3 v1(V(F(i, 1), 0), V(F(i, 1), 1), V(F(i, 1), 2));
        glm::vec3 v2(V(F(i, 2), 0), V(F(i, 2), 1), V(F(i, 2), 2));
        
        float t;
        if (rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t)) {
            if (t < closest_t) {
                closest_t = t;
                hit_triangle = i;
                intersection_point = rayOrigin + rayDir * t;
            }
        }
    }
    
    // If we hit a triangle, find the closest vertex of that triangle
    if (hit_triangle >= 0) {
        float min_dist = std::numeric_limits<float>::max();
        closest_vertex = -1;
        
        // Check only the vertices of the hit triangle
        for (int j = 0; j < 3; j++) {
            int vertex_idx = F(hit_triangle, j);
            glm::vec3 vertex(V(vertex_idx, 0), V(vertex_idx, 1), V(vertex_idx, 2));
            float dist = glm::distance(vertex, intersection_point);
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_vertex = vertex_idx;
            }
        }
        
        return true;
    }
    
    return false;
}


// Update visualization based on current state
void update() {
    // Update mesh and control points
    if (placing_handles) {
        // Show original mesh with control points
        polyscope::getSurfaceMesh("Mesh")->updateVertexPositions(V);

        // Update control points display
        if (polyscope::hasPointCloud("ControlPoints")) {
            polyscope::removePointCloud("ControlPoints");
        }

        if (CV.rows() > 0) {
            std::vector<glm::vec3> points;
            for (int i = 0; i < CV.rows(); i++) {
                points.push_back(glm::vec3(CV(i, 0), CV(i, 1), CV(i, 2)));
            }
            polyscope::registerPointCloud("ControlPoints", points)
                ->setPointRadius(vertexRadius)
                ->setPointColor(glm::vec3(1.0f, 0.7f, 0.2f)); // Orange
        }
    } else {
        // Solve for deformation
        switch (method) {
            case BIHARMONIC: {
                Eigen::MatrixXd D;
                biharmonic_solve(biharmonic_data, CU - CV, D);
                U = V + D;
                break;
            }
            case ARAP: {
                arap_single_iteration(arap_data, arap_K, CU, U);
                break;
            }
        }

        // Update deformed mesh
        polyscope::getSurfaceMesh("Mesh")->updateVertexPositions(U);

        // Update control points display
        if (polyscope::hasPointCloud("ControlPoints")) {
            polyscope::removePointCloud("ControlPoints");
        }

        if (CU.rows() > 0) {
            std::vector<glm::vec3> points;
            for (int i = 0; i < CU.rows(); i++) {
                points.push_back(glm::vec3(CU(i, 0), CU(i, 1), CU(i, 2)));
            }

            glm::vec3 color = (method == BIHARMONIC) ?
                              glm::vec3(0.2f, 0.3f, 0.8f) : // Blue for Biharmonic
                              glm::vec3(0.2f, 0.6f, 0.3f);  // Green for ARAP

            polyscope::registerPointCloud("ControlPoints", points)
                ->setPointRadius(vertexRadius)
                ->setPointColor(color);
        }
    }

    polyscope::requestRedraw();
}

// Main UI callback
void userCallback() {
    ImGui::Text("Mesh Deformation");
    ImGui::Separator();

    // Mode selection
    if (ImGui::Button(placing_handles ? "Switch to Deformation Mode" : "Switch to Handle Placement")) {
        // Check if there are any control points before switching to deformation mode
        if (placing_handles && CV.rows() == 0) {
            ImGui::OpenPopup("Warning");
        } else {
            placing_handles ^= 1;
            if (!placing_handles && CV.rows() > 0) {
                // Switching to deformation mode
                CU = CV;

                // Create list of handle indices
                Eigen::VectorXi b(CV.rows());
                for (int i = 0; i < CV.rows(); i++) {
                    double min_dist = std::numeric_limits<double>::max();
                    int closest_idx = -1;
                    for (int j = 0; j < V.rows(); j++) {
                        double dist = (CV.row(i) - V.row(j)).norm();
                        if (dist < min_dist) {
                            min_dist = dist;
                            closest_idx = j;
                        }
                    }
                    b(i) = closest_idx;
                }

                // Precompute for deformation
                biharmonic_precompute(V, F, b, biharmonic_data);
                arap_precompute(V, F, b, arap_data, arap_K);

                // Lock camera when switching to deformation mode
                savedViewMatrix = polyscope::view::getCameraViewMatrix();
                cameraLocked = true;
            } else if (placing_handles) {
                // Release camera lock when switching to handle placement
                cameraLocked = false;
            }
            update();
        }
    }

    if (ImGui::BeginPopup("Warning")) {
        ImGui::Text("Please place at least one control point first!");
        ImGui::EndPopup();
    }

    // Method selection (only in deformation mode)
    if (!placing_handles) {
        ImGui::Separator();
        ImGui::Text("Deformation Method:");
        bool is_biharmonic = (method == BIHARMONIC);
        bool is_arap = (method == ARAP);

        if (ImGui::RadioButton("Biharmonic", is_biharmonic)) {
            method = BIHARMONIC;
            update();
        }

        if (ImGui::RadioButton("ARAP", is_arap)) {
            method = ARAP;
            update();
        }

        // Add camera lock checkbox
        ImGui::Separator();

        if (ImGui::Checkbox("Lock Camera", &cameraLocked)) {
            if (cameraLocked) {
                // Save current view matrix when locking
                savedViewMatrix = polyscope::view::getCameraViewMatrix();
            }
            update(); // Update to apply camera changes
        }
    }

    // Reset button
    ImGui::Separator();
    if (ImGui::Button("Reset")) {
        // Clear all control points
        CV.resize(0, 3);
        CU.resize(0, 3);

        // Reset mesh to original state
        U = V;

        // Go back to handle placement mode
        placing_handles = true;
        cameraLocked = false;

        update();
    }

    ImGui::End(); // End the default window

    // Handle mouse input for interaction
    if (cameraLocked) {
        polyscope::view::setCameraViewMatrix(savedViewMatrix);
    }

    // Handle selection of control points in placing_handles mode
    if (placing_handles) {
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
            ImVec2 mousePos = ImGui::GetMousePos();
            int closest_vertex;
            if (find_closest_vertex(mousePos, closest_vertex)) {
                // Check if vertex is already a control point
                bool is_new = true;
                Eigen::RowVector3d vertex_pos = V.row(closest_vertex);

                for (int i = 0; i < CV.rows() && is_new; i++) {
                    if ((CV.row(i) - vertex_pos).norm() < 1e-5) {
                        is_new = false;
                    }
                }
                if (is_new) {
                    CV.conservativeResize(CV.rows() + 1, 3);
                    CV.row(CV.rows() - 1) = vertex_pos;
                    update();
                }
            }
        }
    }
    // Handle control point dragging in deformation mode
    else {
        // Mouse down - select closest control point
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse && sel == -1) {
            ImVec2 mousePos = ImGui::GetMousePos();
            last_mouse = mousePos;

            float minScreenDist = 20.0f; // Pixel threshold
            sel = -1;

            // Find closest control point to mouse in screen space
            for (int i = 0; i < CU.rows(); i++) {
                glm::vec3 pos(CU(i, 0), CU(i, 1), CU(i, 2));
                glm::vec2 screenPos = worldToScreen(pos);
                glm::vec2 diff = glm::vec2(mousePos.x, mousePos.y) - screenPos;
                float dist = glm::length(diff);

                if (dist < minScreenDist) {
                    minScreenDist = dist;
                    sel = i;
                }
            }
        }

        // Mouse drag - update control point position
        if (sel != -1 && ImGui::IsMouseDragging(0) && !ImGui::GetIO().WantCaptureMouse) {
            ImVec2 mousePos = ImGui::GetMousePos();

            // Define drag plane - use a plane parallel to the view
            auto center = ImVec2(ImGui::GetIO().DisplaySize.x / 2.0f, ImGui::GetIO().DisplaySize.y / 2.0f);
            auto [dummyOrigin, viewDir] = unproject(center);
            glm::vec3 normal = viewDir; // Plane normal

            // Get the ray from mouse position
            auto [rayOrigin, rayDir] = unproject(mousePos);

            // Use the current position of the control point for the plane
            glm::vec3 planePoint(CU(sel, 0), CU(sel, 1), CU(sel, 2));

            // Find intersection with drag plane
            float denom = glm::dot(rayDir, normal);
            if (std::fabs(denom) > 1e-6) {
                float t = glm::dot(planePoint - rayOrigin, normal) / denom;
                glm::vec3 newPos = rayOrigin + t * rayDir;

                // Update control point position
                CU(sel, 0) = newPos.x;
                CU(sel, 1) = newPos.y;
                CU(sel, 2) = newPos.z;

                // Solve for new deformation
                update();

                last_mouse = mousePos;
            }
        }

        // Mouse up - release control point
        if (ImGui::IsMouseReleased(0) && sel != -1) {
            sel = -1;
        }
    }
}

int main(int argc, char *argv[]) {
    // Load input mesh
    std::string filename = (argc > 1) ? std::string(argv[1]) : "../../data/decimated-knight.off";
    if (!igl::read_triangle_mesh(filename, V, F)) {
        std::cerr << "Failed to load mesh: " << filename << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize Polyscope
    polyscope::init();
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;

    // Register the mesh
    U = V; // Initialize deformed mesh
    polyscope::registerSurfaceMesh("Mesh", V, F);
    // Set mesh to use flat shading instead of smooth shading
    polyscope::getSurfaceMesh("Mesh")->setSmoothShade(false);
    // Make edges visible
    polyscope::getSurfaceMesh("Mesh")->setEdgeWidth(1.0);
    polyscope::getSurfaceMesh("Mesh")->setEdgeColor(glm::vec3(0.0, 0.0, 0.0)); // Black edges


    // Set up user callback
    polyscope::state::userCallback = userCallback;


    // Print instructions
    std::cout << R"(
Mesh Deformation Controls:
[click]        - Place/select control points
[drag]         - Move control points
[Button]       - Toggle between handle placement and deformation
[Radio Button] - Select deformation method (Biharmonic or ARAP)
[Update ARAP]  - Run another iteration of ARAP solver
[Reset All]    - Clear all control points and reset mesh
)" << std::endl;

    // Start the GUI
    polyscope::show();

    return EXIT_SUCCESS;
}
