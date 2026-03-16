// =============================================================================
//  source.cpp  —  WebGL 3D viewer + FEM PDE solver  (Emscripten)
//
//  Coordinate system: X=right, Y=front, Z=up  |  domain +-5, step 0.25
//
//  Subdivision panel  ->  setSubdivision(a,b,N,M,h)
//  PDE solver panel   ->  solvePDE(a,b,N,M,lambda,dirichlet)
//    Solves: -div(eta grad(u)) + lambda*u = f   on [-a,a]x[-b,b]
//    f(x,y) and eta(x,y) evaluated via JS callbacks window._fem_f / window._fem_eta
//
//  v13 fixes vs v12:
//    - BiCGSTAB: variable 'v_vec' persists across iterations holding old A*p
//      (old code wrongly used T = old A*s instead of old A*p in the p update)
//    - Mesh pre-computed once in g_fem, reused by every matvec call
//    - No polygon offset on PDE filled triangles (they live at z=u != 0)
//    - Dirichlet BC support (row replacement method)
//    - Console logging for debug
//
//  Build:
//    emcc src/source.cpp -o app.js \
//         -s USE_WEBGL2=1 -s FULL_ES3=1 -s ALLOW_MEMORY_GROWTH=1 \
//         -s EXPORTED_FUNCTIONS='["_main","_setSubdivision","_solvePDE","_clearPDE"]' \
//         -s EXPORTED_RUNTIME_METHODS='["ccall"]' \
//         -O2
// =============================================================================
#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES2/gl2.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

#include "Camera.h"

// -----------------------------------------------------------------------------
//  GLSL shaders
// -----------------------------------------------------------------------------
static const char* VERT_SRC = R"(
attribute vec3 aPos;
attribute vec3 aColor;
uniform   mat4 uMVP;
varying   vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

static const char* FRAG_SRC = R"(
precision mediump float;
varying vec3 vColor;

void main() {
    gl_FragColor = vec4(vColor, 1.0);
}
)";

// -----------------------------------------------------------------------------
//  Mat4 implementations
// -----------------------------------------------------------------------------
Mat4 mat4Multiply(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            float s = 0.f;
            for (int k = 0; k < 4; k++)
                s += a.m[k*4+row] * b.m[col*4+k];
            r.m[col*4+row] = s;
        }
    return r;
}

Mat4 mat4Perspective(float fovY, float aspect, float znear, float zfar) {
    Mat4 r;
    float f = 1.f / std::tan(fovY * 0.5f);
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.f;
    r.m[14] = (2.f * zfar * znear) / (znear - zfar);
    r.m[15] = 0.f;
    return r;
}

Mat4 mat4LookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = (center - eye).normalize();
    Vec3 s = f.cross(up).normalize();
    Vec3 u = s.cross(f);

    Mat4 r;
    r.m[0]  =  s.x;  r.m[1]  =  u.x;  r.m[2]  = -f.x;  r.m[3]  = 0.f;
    r.m[4]  =  s.y;  r.m[5]  =  u.y;  r.m[6]  = -f.y;  r.m[7]  = 0.f;
    r.m[8]  =  s.z;  r.m[9]  =  u.z;  r.m[10] = -f.z;  r.m[11] = 0.f;
    r.m[12] = -s.dot(eye);
    r.m[13] = -u.dot(eye);
    r.m[14] =  f.dot(eye);
    r.m[15] =  1.f;
    return r;
}

// -----------------------------------------------------------------------------
//  Constants
// -----------------------------------------------------------------------------
static const float AXIS_LEN  = 5.0f;
static const float TICK_STEP = 0.25f;
static const float TICK_SIZE = 0.06f;
static const int   TICK_CNT  = 20;

// -----------------------------------------------------------------------------
//  Global state
// -----------------------------------------------------------------------------
static Camera  g_camera(800.f / 600.f);
static GLuint  g_program    = 0;

static GLuint  g_axesVbo    = 0;
static int     g_axesVtxCnt = 0;

// Subdivision buffers
static GLuint  g_subVbo       = 0;   int g_subVtxCnt    = 0;
static GLuint  g_subTriVbo    = 0;   int g_subTriVtxCnt = 0;
static GLuint  g_baryVbo      = 0;   int g_baryVtxCnt   = 0;

// PDE solution surface buffers
static GLuint  g_pdeTriVbo    = 0;   int g_pdeTriVtxCnt  = 0;
static GLuint  g_pdeWireVbo   = 0;   int g_pdeWireVtxCnt = 0;

static bool    g_mouseDown  = false;
static float   g_lastX      = 0.f;
static float   g_lastY      = 0.f;
static int     g_canvasW    = 800;
static int     g_canvasH    = 600;

// -----------------------------------------------------------------------------
//  Geometry -- Axes
// -----------------------------------------------------------------------------
static void buildAxes() {
    const float L = AXIS_LEN;
    const float T = TICK_SIZE;
    std::vector<float> verts;
    verts.insert(verts.end(), { -L, 0.f, 0.f,  0.40f, 0.f,  0.f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.40f, 0.f,  0.f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.50f, 0.f,  0.f  });
    verts.insert(verts.end(), {   L, 0.f, 0.f, 1.0f,  0.f,  0.f  });
    verts.insert(verts.end(), { 0.f, -L, 0.f,  0.f, 0.40f,  0.f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.f, 0.40f,  0.f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.f, 0.50f,  0.f  });
    verts.insert(verts.end(), { 0.f,   L, 0.f, 0.f, 1.0f,   0.f  });
    verts.insert(verts.end(), { 0.f, 0.f, -L,  0.f, 0.f,  0.40f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.f, 0.f,  0.40f  });
    verts.insert(verts.end(), { 0.f, 0.f, 0.f, 0.f, 0.f,  0.50f  });
    verts.insert(verts.end(), { 0.f, 0.f,   L, 0.f, 0.f,  1.0f   });
    for (int i = -TICK_CNT; i <= TICK_CNT; i++) {
        if (i == 0) continue;
        float v = (float)i * TICK_STEP;
        verts.insert(verts.end(), { v, 0.f, -T,  1.0f, 0.50f, 0.50f });
        verts.insert(verts.end(), { v, 0.f,  T,  1.0f, 0.50f, 0.50f });
        verts.insert(verts.end(), { 0.f, v, -T,  0.50f, 1.0f, 0.50f });
        verts.insert(verts.end(), { 0.f, v,  T,  0.50f, 1.0f, 0.50f });
        verts.insert(verts.end(), { -T, 0.f, v,  0.50f, 0.50f, 1.0f });
        verts.insert(verts.end(), {  T, 0.f, v,  0.50f, 0.50f, 1.0f });
    }
    g_axesVtxCnt = (int)verts.size() / 6;
    glGenBuffers(1, &g_axesVbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_axesVbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(float)),
                 verts.data(), GL_STATIC_DRAW);
}

// -----------------------------------------------------------------------------
//  Geometry -- Subdivision
// -----------------------------------------------------------------------------
static void buildSubdivision(float a, float b, int N, int M, float h) {
    if (g_subVbo)    { glDeleteBuffers(1,&g_subVbo);    g_subVbo    =0; g_subVtxCnt    =0; }
    if (g_subTriVbo) { glDeleteBuffers(1,&g_subTriVbo); g_subTriVbo =0; g_subTriVtxCnt =0; }
    if (g_baryVbo)   { glDeleteBuffers(1,&g_baryVbo);   g_baryVbo   =0; g_baryVtxCnt   =0; }
    if (a<=0.f || b<=0.f || N<=0 || M<=0) return;

    const long long cells = (long long)N*M;
    const float dx = 2.f*a/(float)N;
    const float dy = 2.f*b/(float)M;
    const float WR=0.30f, WG=0.60f, WB=1.0f;
    const float RA=0.10f, GA=0.30f, BA=0.75f;
    const float RB=0.25f, GB=0.55f, BB=0.95f;

    // Flat fills: cheap (2 tris/cell), allow up to 1M cells (~1000x1000)
    // Pyramids (h>0): 9x heavier geometry, cap at 300x300 = 90000 cells
    const long long FLAT_CAP = 1000000LL;
    const long long PYRA_CAP = 90000LL;

    if ((h <= 0.f && cells <= FLAT_CAP) || (h > 0.f && cells <= PYRA_CAP)) {
        if (h <= 0.f) {
            std::vector<float> tris;
            tris.reserve((size_t)(cells*2*3*6));
            for (int j=0; j<M; j++) {
                float y0=-b+(float)j*dy, y1=y0+dy;
                for (int i=0; i<N; i++) {
                    float x0=-a+(float)i*dx, x1=x0+dx;
                    float R=((i+j)%2==0)?RA:RB, G=((i+j)%2==0)?GA:GB, B=((i+j)%2==0)?BA:BB;
                    if ((i+j)%2==0) {
                        tris.insert(tris.end(),{x0,y0,0.f,R,G,B, x1,y0,0.f,R,G,B, x0,y1,0.f,R,G,B});
                        tris.insert(tris.end(),{x1,y0,0.f,R,G,B, x1,y1,0.f,R,G,B, x0,y1,0.f,R,G,B});
                    } else {
                        tris.insert(tris.end(),{x0,y0,0.f,R,G,B, x1,y0,0.f,R,G,B, x1,y1,0.f,R,G,B});
                        tris.insert(tris.end(),{x0,y0,0.f,R,G,B, x1,y1,0.f,R,G,B, x0,y1,0.f,R,G,B});
                    }
                }
            }
            g_subTriVtxCnt=(int)tris.size()/6;
            glGenBuffers(1,&g_subTriVbo);
            glBindBuffer(GL_ARRAY_BUFFER,g_subTriVbo);
            glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(tris.size()*sizeof(float)),tris.data(),GL_DYNAMIC_DRAW);
        } else {
            const float C0r=0.95f,C0g=0.35f,C0b=0.20f;
            const float C1r=0.25f,C1g=0.90f,C1b=0.35f;
            const float C2r=0.20f,C2g=0.65f,C2b=1.00f;
            const float Dk=0.08f;
            std::vector<float> bary;
            bary.reserve((size_t)(cells*2*9*6));
            for (int j=0; j<M; j++) {
                float y0=-b+(float)j*dy, y1=y0+dy;
                for (int i=0; i<N; i++) {
                    float x0=-a+(float)i*dx, x1=x0+dx;
                    float p[2][3][2];
                    if ((i+j)%2==0) {
                        p[0][0][0]=x0;p[0][0][1]=y0; p[0][1][0]=x1;p[0][1][1]=y0; p[0][2][0]=x0;p[0][2][1]=y1;
                        p[1][0][0]=x1;p[1][0][1]=y0; p[1][1][0]=x1;p[1][1][1]=y1; p[1][2][0]=x0;p[1][2][1]=y1;
                    } else {
                        p[0][0][0]=x0;p[0][0][1]=y0; p[0][1][0]=x1;p[0][1][1]=y0; p[0][2][0]=x1;p[0][2][1]=y1;
                        p[1][0][0]=x0;p[1][0][1]=y0; p[1][1][0]=x1;p[1][1][1]=y1; p[1][2][0]=x0;p[1][2][1]=y1;
                    }
                    for (int t=0; t<2; t++) {
                        float ax=p[t][0][0],ay=p[t][0][1];
                        float bx=p[t][1][0],by_=p[t][1][1];
                        float cx=p[t][2][0],cy=p[t][2][1];
                        bary.insert(bary.end(),{ax,ay,h,C0r,C0g,C0b, bx,by_,0.f,Dk,Dk,Dk, cx,cy,0.f,Dk,Dk,Dk});
                        bary.insert(bary.end(),{ax,ay,0.f,Dk,Dk,Dk, bx,by_,h,C1r,C1g,C1b, cx,cy,0.f,Dk,Dk,Dk});
                        bary.insert(bary.end(),{ax,ay,0.f,Dk,Dk,Dk, bx,by_,0.f,Dk,Dk,Dk, cx,cy,h,C2r,C2g,C2b});
                    }
                }
            }
            g_baryVtxCnt=(int)bary.size()/6;
            glGenBuffers(1,&g_baryVbo);
            glBindBuffer(GL_ARRAY_BUFFER,g_baryVbo);
            glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(bary.size()*sizeof(float)),bary.data(),GL_DYNAMIC_DRAW);
        }
    }

    std::vector<float> lines;
    for (int i=0; i<=N; i++) {
        float x=-a+(float)i*dx;
        lines.insert(lines.end(),{x,-b,0.f,WR,WG,WB}); lines.insert(lines.end(),{x,b,0.f,WR,WG,WB});
    }
    for (int j=0; j<=M; j++) {
        float y=-b+(float)j*dy;
        lines.insert(lines.end(),{-a,y,0.f,WR,WG,WB}); lines.insert(lines.end(),{a,y,0.f,WR,WG,WB});
    }
    if (cells <= 1000000LL) {
        for (int j=0; j<M; j++) {
            float y0=-b+(float)j*dy, y1=y0+dy;
            for (int i=0; i<N; i++) {
                float x0=-a+(float)i*dx, x1=x0+dx;
                if ((i+j)%2==0) {
                    lines.insert(lines.end(),{x0,y1,0.f,WR,WG,WB}); lines.insert(lines.end(),{x1,y0,0.f,WR,WG,WB});
                } else {
                    lines.insert(lines.end(),{x0,y0,0.f,WR,WG,WB}); lines.insert(lines.end(),{x1,y1,0.f,WR,WG,WB});
                }
            }
        }
        if (h>0.f) {
            const float SR=0.55f,SG=0.55f,SB=0.55f;
            for (int j=0; j<=M; j++) {
                float y=-b+(float)j*dy;
                for (int i=0; i<=N; i++) {
                    float x=-a+(float)i*dx;
                    lines.insert(lines.end(),{x,y,0.f,SR,SG,SB}); lines.insert(lines.end(),{x,y,h,SR,SG,SB});
                }
            }
        }
    }
    g_subVtxCnt=(int)lines.size()/6;
    glGenBuffers(1,&g_subVbo);
    glBindBuffer(GL_ARRAY_BUFFER,g_subVbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(lines.size()*sizeof(float)),lines.data(),GL_DYNAMIC_DRAW);
}

// =============================================================================
//  JS callbacks
// =============================================================================

EM_JS(double, js_eval_eta, (double x, double y), {
    if (window._fem_eta) {
        try { return window._fem_eta(x, y); } catch(e) { return 1.0; }
    }
    return 1.0;
});

EM_JS(double, js_eval_f, (double x, double y), {
    if (window._fem_f) {
        try { return window._fem_f(x, y); } catch(e) { return 0.0; }
    }
    return 0.0;
});

EM_JS(void, js_setPDEStatus, (const char* msg), {
    var el = document.getElementById('pde-status');
    if (el) el.innerHTML = UTF8ToString(msg);
});

EM_JS(void, js_log, (const char* msg), {
    console.log('[FEM] ' + UTF8ToString(msg));
});

// =============================================================================
//  FEM State -- pre-computed mesh, reused across matvec calls
// =============================================================================
struct FEMState {
    int N = 0, M = 0;
    double a = 0, b = 0, lam = 0;
    bool dirichlet = true;
    std::vector<double> Xc, Yc;
    std::vector<std::array<int,3>> tri;

    void build(int n, int m, double a_, double b_, double lam_, bool dir) {
        N=n; M=m; a=a_; b=b_; lam=lam_; dirichlet=dir;
        Xc.resize(N+1);
        double hx = 2.0*a/N;
        for (int i=0; i<=N; i++) Xc[i] = i*hx - a;
        Yc.resize(M+1);
        double hy = 2.0*b/M;
        for (int j=0; j<=M; j++) Yc[j] = j*hy - b;
        int K = 2*N*M;
        tri.resize(K);
        int k=0;
        for (int j=0; j<M; j++) for (int i=0; i<N; i++) {
            int A=(N+1)*j+i, B=(N+1)*j+(i+1), C=(N+1)*(j+1)+i, D=(N+1)*(j+1)+(i+1);
            tri[k++] = {{A,B,D}};
            tri[k++] = {{A,D,C}};
        }
    }

    bool onBoundary(int s) const {
        int ii = s%(N+1), jj = s/(N+1);
        return (ii==0 || ii==N || jj==0 || jj==M);
    }
};
static FEMState g_fem;

// =============================================================================
//  FEM element routines
// =============================================================================

static void calcBT(const double* tx, const double* ty,
                   double& b00, double& b01, double& b10, double& b11) {
    b00=tx[1]-tx[0]; b01=tx[2]-tx[0];
    b10=ty[1]-ty[0]; b11=ty[2]-ty[0];
}

// Approximation of integral_T eta dA  (midpoint rule at 3 edge midpoints)
static double integEta(const double* tx, const double* ty) {
    double b00,b01,b10,b11;
    calcBT(tx,ty,b00,b01,b10,b11);
    double absdet = fabs(b00*b11 - b01*b10);
    double m01x=0.5*(tx[0]+tx[1]), m01y=0.5*(ty[0]+ty[1]);
    double m12x=0.5*(tx[1]+tx[2]), m12y=0.5*(ty[1]+ty[2]);
    double m02x=0.5*(tx[0]+tx[2]), m02y=0.5*(ty[0]+ty[2]);
    double avg = (js_eval_eta(m01x,m01y)+js_eval_eta(m12x,m12y)+js_eval_eta(m02x,m02y))/3.0;
    return avg * absdet / 2.0;
}

// Packed index for 3x3 symmetric matrix (upper triangle)
static int symIdx(int i, int j) {
    if (i>j){int t=i;i=j;j=t;}
    if(i==0&&j==0)return 0; if(i==0&&j==1)return 1; if(i==0&&j==2)return 2;
    if(i==1&&j==1)return 3; if(i==1&&j==2)return 4; return 5;
}

// Diffusion element matrix K[p] = integral_T eta grad(phi_i).grad(phi_j) dA
static void diffTerm(const double* tx, const double* ty, double K[6]) {
    double b00,b01,b10,b11;
    calcBT(tx,ty,b00,b01,b10,b11);
    double det = b00*b11 - b01*b10;
    if (fabs(det) < 1e-30) { for(int i=0;i<6;i++) K[i]=0; return; }
    double val = integEta(tx,ty);
    double inv00= b11/det, inv01=-b10/det;
    double inv10=-b01/det, inv11= b00/det;
    // Reference gradients of P1 hat functions: phi0=(-1,-1), phi1=(1,0), phi2=(0,1)
    static const double gl[3][2] = {{-1,-1},{1,0},{0,1}};
    double gr[3][2];
    for (int i=0; i<3; i++) {
        gr[i][0] = inv00*gl[i][0] + inv01*gl[i][1];
        gr[i][1] = inv10*gl[i][0] + inv11*gl[i][1];
    }
    int p=0;
    for (int i=0; i<3; i++) for (int j=i; j<3; j++)
        K[p++] = (gr[i][0]*gr[j][0] + gr[i][1]*gr[j][1]) * val;
}

// Reaction element matrix M[p] = lambda * integral_T phi_i*phi_j dA
static void reacTerm(const double* tx, const double* ty, double lam, double Ml[6]) {
    double b00,b01,b10,b11;
    calcBT(tx,ty,b00,b01,b10,b11);
    double absdet = fabs(b00*b11 - b01*b10);
    int p=0;
    for (int i=0; i<3; i++) for (int j=i; j<3; j++)
        Ml[p++] = absdet * lam * (i==j ? 1.0/12.0 : 1.0/24.0);
}

// =============================================================================
//  Matrix-vector product W = A * V
//  Uses pre-computed g_fem mesh.
//  Dirichlet BCs: boundary rows become identity (W[k] = V[k]).
// =============================================================================
static std::vector<double> fem_matvec(const std::vector<double>& V) {
    int G = (g_fem.N+1)*(g_fem.M+1);
    std::vector<double> W(G, 0.0);

    for (const auto& t : g_fem.tri) {
        int s[3] = {t[0], t[1], t[2]};
        double tx[3], ty[3];
        for (int k=0; k<3; k++) {
            int ii=s[k]%(g_fem.N+1), jj=s[k]/(g_fem.N+1);
            tx[k]=g_fem.Xc[ii]; ty[k]=g_fem.Yc[jj];
        }
        double KD[6], KR[6];
        diffTerm(tx,ty,KD);
        reacTerm(tx,ty,g_fem.lam,KR);
        for (int i=0; i<3; i++) {
            double res=0.0;
            for (int j=0; j<3; j++)
                res += V[s[j]] * (KD[symIdx(i,j)] + KR[symIdx(i,j)]);
            W[s[i]] += res;
        }
    }

    // Dirichlet BC: replace boundary rows with identity
    if (g_fem.dirichlet) {
        for (int s=0; s<G; s++) {
            if (g_fem.onBoundary(s)) W[s] = V[s];
        }
    }
    return W;
}

// =============================================================================
//  Right-hand side  B[k] = integral_Omega f * phi_k dA
//  Midpoint quadrature at 3 edge midpoints per triangle.
//  Dirichlet: B[k] = 0 on boundary.
// =============================================================================
static std::vector<double> fem_rhs() {
    int G = (g_fem.N+1)*(g_fem.M+1);
    std::vector<double> B(G, 0.0);

    for (const auto& t : g_fem.tri) {
        int s[3] = {t[0], t[1], t[2]};
        double tx[3], ty[3];
        for (int k=0; k<3; k++) {
            int ii=s[k]%(g_fem.N+1), jj=s[k]/(g_fem.N+1);
            tx[k]=g_fem.Xc[ii]; ty[k]=g_fem.Yc[jj];
        }
        double b00,b01,b10,b11;
        calcBT(tx,ty,b00,b01,b10,b11);
        double aire = fabs(b00*b11 - b01*b10) / 2.0;
        double m01x=(tx[0]+tx[1])/2, m01y=(ty[0]+ty[1])/2;
        double m12x=(tx[1]+tx[2])/2, m12y=(ty[1]+ty[2])/2;
        double m02x=(tx[0]+tx[2])/2, m02y=(ty[0]+ty[2])/2;
        double f01=js_eval_f(m01x,m01y), f12=js_eval_f(m12x,m12y), f02=js_eval_f(m02x,m02y);
        // phi_0 nonzero on edges 0-1 (val=1/2) and 0-2 (val=1/2), zero on 1-2
        // phi_1 nonzero on edges 0-1 (val=1/2) and 1-2 (val=1/2), zero on 0-2
        // phi_2 nonzero on edges 1-2 (val=1/2) and 0-2 (val=1/2), zero on 0-1
        B[s[0]] += aire * (f01 + f02) / 6.0;
        B[s[1]] += aire * (f01 + f12) / 6.0;
        B[s[2]] += aire * (f12 + f02) / 6.0;
    }

    if (g_fem.dirichlet) {
        int G2 = (g_fem.N+1)*(g_fem.M+1);
        for (int s=0; s<G2; s++)
            if (g_fem.onBoundary(s)) B[s] = 0.0;
    }
    return B;
}

// =============================================================================
//  BiCGSTAB  --  solves A x = b  using pre-computed g_fem
//
//  KEY FIX vs v12: 'v_vec' persists across iterations, always holding the
//  CURRENT A*p from the previous call to v_vec = fem_matvec(p).
//  The p-update formula needs the OLD A*p, which is exactly what v_vec holds
//  at the start of each iteration (before we overwrite it).
//
//  Old v12 code used T (= old A*s) which was WRONG.
// =============================================================================
static std::vector<double> fem_bicgstab(const std::vector<double>& b_rhs,
                                         double tol, int maxIter)
{
    int n = (int)b_rhs.size();
    std::vector<double> x(n, 0.0);
    std::vector<double> r(b_rhs);       // r = b - A*x0 = b  (x0 = 0)
    std::vector<double> r_hat(b_rhs);   // shadow residual, arbitrary; standard: r_hat = r0

    double normb = 0;
    for (int i=0; i<n; i++) normb += b_rhs[i]*b_rhs[i];
    normb = sqrt(normb);
    if (normb < 1e-30) {
        js_log("BiCGSTAB: RHS norm is zero => trivial solution");
        return x;
    }

    // v_vec persists: always holds the most recently computed A*p
    std::vector<double> p(n, 0.0), v_vec(n, 0.0);
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {

        double rho = 0;
        for (int i=0; i<n; i++) rho += r_hat[i]*r[i];
        if (fabs(rho) < 1e-300) { js_log("BiCGSTAB: rho breakdown"); break; }

        // Update p.
        // iter==0: beta=0 => p = r  (v_vec is zero, doesn't matter)
        // iter >0: v_vec is the OLD A*p from the previous iteration -- CORRECT
        double beta = (iter == 0) ? 0.0 : (rho / rho_old) * (alpha / omega);
        for (int i=0; i<n; i++)
            p[i] = r[i] + beta * (p[i] - omega * v_vec[i]);

        // Compute v_vec = A*p  (overwrites v_vec; will be used as "old A*p" next iteration)
        v_vec = fem_matvec(p);

        double rv = 0;
        for (int i=0; i<n; i++) rv += r_hat[i]*v_vec[i];
        if (fabs(rv) < 1e-300) { js_log("BiCGSTAB: <r_hat,v> breakdown"); break; }
        alpha = rho / rv;

        // s = r - alpha * v_vec
        std::vector<double> s_vec(n);
        for (int i=0; i<n; i++) s_vec[i] = r[i] - alpha * v_vec[i];

        double norms = 0;
        for (int i=0; i<n; i++) norms += s_vec[i]*s_vec[i];
        if (sqrt(norms) < tol * normb) {
            for (int i=0; i<n; i++) x[i] += alpha * p[i];
            char msg[80]; snprintf(msg,80,"BiCGSTAB converged (s-check) iter=%d",iter+1);
            js_log(msg);
            return x;
        }

        // t = A*s
        auto t_vec = fem_matvec(s_vec);

        double ts=0, tt=0;
        for (int i=0; i<n; i++) { ts += t_vec[i]*s_vec[i]; tt += t_vec[i]*t_vec[i]; }
        if (fabs(tt) < 1e-300) { js_log("BiCGSTAB: tt breakdown"); break; }
        omega = ts / tt;

        for (int i=0; i<n; i++) x[i] += alpha*p[i] + omega*s_vec[i];
        for (int i=0; i<n; i++) r[i]  = s_vec[i] - omega*t_vec[i];

        double normr = 0;
        for (int i=0; i<n; i++) normr += r[i]*r[i];
        if (sqrt(normr) < tol * normb) {
            char msg[80]; snprintf(msg,80,"BiCGSTAB converged iter=%d",iter+1);
            js_log(msg);
            return x;
        }

        rho_old = rho;
    }
    js_log("BiCGSTAB: reached max iterations");
    return x;
}

// =============================================================================
//  PDE Surface Geometry -- vertices at (x_k, y_k, X[k])
//  Each node height = FEM coefficient = u_h(node k)
//  Together the mesh represents u_h(x,y) = sum_k X[k] * phi_k(x,y)
// =============================================================================

// Blue -> Cyan -> Green -> Yellow -> Red heatmap
static void fem_colormap(float t, float& r, float& g, float& b) {
    t = std::max(0.f, std::min(1.f, t));
    if      (t < 0.25f) { float s=t/0.25f;          r=0.f; g=s;     b=1.f;   }
    else if (t < 0.50f) { float s=(t-0.25f)/0.25f;  r=0.f; g=1.f;   b=1.f-s; }
    else if (t < 0.75f) { float s=(t-0.50f)/0.25f;  r=s;   g=1.f;   b=0.f;   }
    else                { float s=(t-0.75f)/0.25f;  r=1.f; g=1.f-s; b=0.f;   }
}

static void buildPDESurface(const std::vector<double>& sol) {
    if (g_pdeTriVbo)  { glDeleteBuffers(1,&g_pdeTriVbo);  g_pdeTriVbo =0; g_pdeTriVtxCnt =0; }
    if (g_pdeWireVbo) { glDeleteBuffers(1,&g_pdeWireVbo); g_pdeWireVbo=0; g_pdeWireVtxCnt=0; }

    double umin = *std::min_element(sol.begin(), sol.end());
    double umax = *std::max_element(sol.begin(), sol.end());
    double urange = (umax > umin + 1e-12) ? (umax - umin) : 1.0;

    char dbg[128];
    snprintf(dbg, sizeof(dbg), "buildPDESurface: u=[%.4g, %.4g], %d nodes, %d tris",
             umin, umax, (int)sol.size(), (int)g_fem.tri.size());
    js_log(dbg);

    std::vector<float> tris, lines;
    tris.reserve(g_fem.tri.size() * 3 * 6);
    lines.reserve(g_fem.tri.size() * 6 * 6);

    for (const auto& t : g_fem.tri) {
        int s[3] = {t[0], t[1], t[2]};
        float vx[3], vy[3], vz[3], vr[3], vg[3], vb_[3];
        for (int k=0; k<3; k++) {
            int ii=s[k]%(g_fem.N+1), jj=s[k]/(g_fem.N+1);
            vx[k] = (float)g_fem.Xc[ii];
            vy[k] = (float)g_fem.Yc[jj];
            vz[k] = (float)sol[s[k]];   // height = FEM coefficient X[k]
            fem_colormap((float)((sol[s[k]]-umin)/urange), vr[k], vg[k], vb_[k]);
        }
        // Filled triangle: 3 vertices, each with position + color
        for (int k=0; k<3; k++)
            tris.insert(tris.end(), {vx[k], vy[k], vz[k], vr[k], vg[k], vb_[k]});
        // Wireframe edges (darker)
        const float dk = 0.6f;
        for (int e=0; e<3; e++) {
            int e2=(e+1)%3;
            lines.insert(lines.end(),{
                vx[e],  vy[e],  vz[e],  vr[e]*dk,  vg[e]*dk,  vb_[e]*dk,
                vx[e2], vy[e2], vz[e2], vr[e2]*dk, vg[e2]*dk, vb_[e2]*dk
            });
        }
    }

    g_pdeTriVtxCnt = (int)tris.size()/6;
    glGenBuffers(1,&g_pdeTriVbo);
    glBindBuffer(GL_ARRAY_BUFFER,g_pdeTriVbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(tris.size()*sizeof(float)),tris.data(),GL_DYNAMIC_DRAW);

    g_pdeWireVtxCnt = (int)lines.size()/6;
    glGenBuffers(1,&g_pdeWireVbo);
    glBindBuffer(GL_ARRAY_BUFFER,g_pdeWireVbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(lines.size()*sizeof(float)),lines.data(),GL_DYNAMIC_DRAW);
}

// =============================================================================
//  Exported functions
// =============================================================================
extern "C" {

    EMSCRIPTEN_KEEPALIVE
    void setSubdivision(float a, float b, int N, int M, float h) {
        buildSubdivision(a, b, N, M, h);
    }

    // solvePDE: called from JS after setting window._fem_f and window._fem_eta
    // dirichlet=1: homogeneous Dirichlet BCs (u=0 on boundary)
    // dirichlet=0: homogeneous Neumann BCs (natural; requires lambda>0)
    EMSCRIPTEN_KEEPALIVE
    void solvePDE(float a, float b, int N, int M, float lambda, int dirichlet) {
        if (g_pdeTriVbo)  { glDeleteBuffers(1,&g_pdeTriVbo);  g_pdeTriVbo =0; g_pdeTriVtxCnt =0; }
        if (g_pdeWireVbo) { glDeleteBuffers(1,&g_pdeWireVbo); g_pdeWireVbo=0; g_pdeWireVtxCnt=0; }

        if (a<=0.f || b<=0.f || N<=0 || M<=0) {
            js_setPDEStatus("Cleared.");
            return;
        }

        double lam = (double)lambda;
        // Neumann requires lambda>0 to avoid singular system
        if (!dirichlet && lam < 1e-6) lam = 1e-6;

        char msg[128];
        snprintf(msg, sizeof(msg),
            "solvePDE: a=%.2f b=%.2f N=%d M=%d lam=%.4g %s",
            (double)a,(double)b,N,M,lam,dirichlet?"Dirichlet":"Neumann");
        js_log(msg);

        js_setPDEStatus("<span style='color:#80d0ff'>Building mesh...</span>");
        // Pre-compute mesh once -- reused by every matvec call in BiCGSTAB
        g_fem.build(N, M, (double)a, (double)b, lam, (bool)dirichlet);

        js_setPDEStatus("<span style='color:#80d0ff'>Assembling RHS...</span>");
        auto B = fem_rhs();

        js_setPDEStatus("<span style='color:#80d0ff'>Running BiCGSTAB...</span>");
        auto sol = fem_bicgstab(B, 1e-8, 2000);

        js_setPDEStatus("<span style='color:#80d0ff'>Building surface...</span>");
        buildPDESurface(sol);

        double umin = *std::min_element(sol.begin(), sol.end());
        double umax = *std::max_element(sol.begin(), sol.end());
        snprintf(msg, sizeof(msg),
            "<b style='color:#80e8a0'>Solved OK</b> &nbsp;%s<br>"
            "%d nodes &nbsp;&middot;&nbsp; %d triangles<br>"
            "u &isin; [%.4g,&thinsp;%.4g]",
            dirichlet ? "Dirichlet" : "Neumann",
            (N+1)*(M+1), 2*N*M, umin, umax);
        js_setPDEStatus(msg);
    }

    EMSCRIPTEN_KEEPALIVE
    void clearPDE() {
        if (g_pdeTriVbo)  { glDeleteBuffers(1,&g_pdeTriVbo);  g_pdeTriVbo =0; g_pdeTriVtxCnt =0; }
        if (g_pdeWireVbo) { glDeleteBuffers(1,&g_pdeWireVbo); g_pdeWireVbo=0; g_pdeWireVtxCnt=0; }
        js_setPDEStatus("Cleared.");
    }
}

// -----------------------------------------------------------------------------
//  Shader helpers
// -----------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        printf("[shader error] %s\n", log);
    }
    return s;
}

// -----------------------------------------------------------------------------
//  Draw helper
// -----------------------------------------------------------------------------
static void drawBuffer(GLuint vbo, GLenum mode, int count) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GLint posLoc = glGetAttribLocation(g_program, "aPos");
    GLint colLoc = glGetAttribLocation(g_program, "aColor");
    glEnableVertexAttribArray(posLoc);
    glEnableVertexAttribArray(colLoc);
    const int stride = 6 * sizeof(float);
    glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glVertexAttribPointer(colLoc, 3, GL_FLOAT, GL_FALSE, stride,
                          (void*)(3 * sizeof(float)));
    glDrawArrays(mode, 0, count);
}

// -----------------------------------------------------------------------------
//  3D -> 2D screen projection
// -----------------------------------------------------------------------------
static bool projectToScreen(Vec3 p, float& sx, float& sy) {
    Mat4 mvp = mat4Multiply(g_camera.getProjectionMatrix(),
                            g_camera.getViewMatrix());
    float cx = mvp.m[0]*p.x + mvp.m[4]*p.y + mvp.m[8]*p.z  + mvp.m[12];
    float cy = mvp.m[1]*p.x + mvp.m[5]*p.y + mvp.m[9]*p.z  + mvp.m[13];
    float cw = mvp.m[3]*p.x + mvp.m[7]*p.y + mvp.m[11]*p.z + mvp.m[15];
    if (cw <= 0.001f) return false;
    sx = ( cx/cw*0.5f + 0.5f)         * (float)g_canvasW;
    sy = (1.f - (cy/cw*0.5f + 0.5f))  * (float)g_canvasH;
    return true;
}

// -----------------------------------------------------------------------------
//  EM_JS -- 2-D label overlay
// -----------------------------------------------------------------------------
EM_JS(void, js_clearLabels, (), {
    var c = document.getElementById('labelCanvas');
    if (!c) return;
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
});

EM_JS(void, js_drawLabel,
    (double sx, double sy, const char* txt, float r, float g, float b, int bold),
{
    var c = document.getElementById('labelCanvas');
    if (!c) return;
    var ctx = c.getContext('2d');
    var str = UTF8ToString(txt);
    ctx.font = bold ? 'bold 15px monospace' : '10px monospace';
    ctx.fillStyle = 'rgba('+Math.round(r*255)+','+Math.round(g*255)+','+Math.round(b*255)+',0.90)';
    ctx.fillText(str, sx+4, sy-3);
});

// -----------------------------------------------------------------------------
//  Draw all axis labels
// -----------------------------------------------------------------------------
static void drawLabels() {
    js_clearLabels();
    char buf[16];
    float sx, sy;

    for (int i=-TICK_CNT; i<=TICK_CNT; i++) {
        if (i==0) continue;
        float v = (float)i * TICK_STEP;
        if (projectToScreen({v,0.f,0.f}, sx, sy)) {
            if (i%4==0) snprintf(buf,sizeof(buf),"%.0f",(double)v);
            else        snprintf(buf,sizeof(buf),"%.2f",(double)v);
            js_drawLabel((double)sx,(double)sy,buf, 1.f,0.45f,0.45f, 0);
        }
    }
    if (projectToScreen({AXIS_LEN+0.4f,0.f,0.f}, sx, sy))
        js_drawLabel((double)sx,(double)sy,"X", 1.f,0.15f,0.15f, 1);

    for (int i=-TICK_CNT; i<=TICK_CNT; i++) {
        if (i==0) continue;
        float v = (float)i * TICK_STEP;
        if (projectToScreen({0.f,v,0.f}, sx, sy)) {
            if (i%4==0) snprintf(buf,sizeof(buf),"%.0f",(double)v);
            else        snprintf(buf,sizeof(buf),"%.2f",(double)v);
            js_drawLabel((double)sx,(double)sy,buf, 0.45f,1.f,0.45f, 0);
        }
    }
    if (projectToScreen({0.f,AXIS_LEN+0.4f,0.f}, sx, sy))
        js_drawLabel((double)sx,(double)sy,"Y", 0.15f,1.f,0.15f, 1);

    for (int i=-TICK_CNT; i<=TICK_CNT; i++) {
        if (i==0) continue;
        float v = (float)i * TICK_STEP;
        if (projectToScreen({0.f,0.f,v}, sx, sy)) {
            if (i%4==0) snprintf(buf,sizeof(buf),"%.0f",(double)v);
            else        snprintf(buf,sizeof(buf),"%.2f",(double)v);
            js_drawLabel((double)sx,(double)sy,buf, 0.45f,0.6f,1.f, 0);
        }
    }
    if (projectToScreen({0.f,0.f,AXIS_LEN+0.4f}, sx, sy))
        js_drawLabel((double)sx,(double)sy,"Z", 0.15f,0.4f,1.f, 1);
}

// -----------------------------------------------------------------------------
//  Input callbacks
// -----------------------------------------------------------------------------
static EM_BOOL onMouseDown(int, const EmscriptenMouseEvent* e, void*) {
    g_mouseDown = true;
    g_lastX = (float)e->clientX; g_lastY = (float)e->clientY;
    return EM_TRUE;
}
static EM_BOOL onMouseUp(int, const EmscriptenMouseEvent*, void*) {
    g_mouseDown = false; return EM_TRUE;
}
static EM_BOOL onMouseMove(int, const EmscriptenMouseEvent* e, void*) {
    if (!g_mouseDown) return EM_FALSE;
    float dx = (float)e->clientX - g_lastX;
    float dy = (float)e->clientY - g_lastY;
    g_camera.orbit(-dx*0.008f, -dy*0.008f);
    g_lastX=(float)e->clientX; g_lastY=(float)e->clientY;
    return EM_TRUE;
}
static EM_BOOL onWheel(int, const EmscriptenWheelEvent* e, void*) {
    float delta = (float)e->deltaY;
    if (e->deltaMode==1) delta*=16.f;
    if (e->deltaMode==2) delta*=400.f;
    g_camera.zoom(delta*0.01f);
    return EM_TRUE;
}

// -----------------------------------------------------------------------------
//  Render loop
// -----------------------------------------------------------------------------
static void mainLoop() {
    glViewport(0, 0, g_canvasW, g_canvasH);
    glClearColor(0.08f, 0.08f, 0.12f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_program);

    Mat4 view = g_camera.getViewMatrix();
    Mat4 proj = g_camera.getProjectionMatrix();
    Mat4 mvp  = mat4Multiply(proj, view);
    GLint mvpLoc = glGetUniformLocation(g_program, "uMVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp.m);

    // 1 -- Flat fill (subdivision, h==0)
    if (g_subTriVbo && g_subTriVtxCnt > 0) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.f, 1.f);
        drawBuffer(g_subTriVbo, GL_TRIANGLES, g_subTriVtxCnt);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    // 2 -- Barycentric pyramid surface (subdivision, h>0)
    if (g_baryVbo && g_baryVtxCnt > 0) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.f, 1.f);
        drawBuffer(g_baryVbo, GL_TRIANGLES, g_baryVtxCnt);
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    // 3 -- Subdivision wireframe
    if (g_subVbo && g_subVtxCnt > 0)
        drawBuffer(g_subVbo, GL_LINES, g_subVtxCnt);

    // 4 -- PDE solution filled triangles
    //      NO polygon offset: the surface lives at z=u(x,y) which is nonzero,
    //      so there is no z-fighting with the flat subdivision at z=0.
    if (g_pdeTriVbo && g_pdeTriVtxCnt > 0)
        drawBuffer(g_pdeTriVbo, GL_TRIANGLES, g_pdeTriVtxCnt);

    // 5 -- PDE mesh wireframe on top of filled surface
    if (g_pdeWireVbo && g_pdeWireVtxCnt > 0)
        drawBuffer(g_pdeWireVbo, GL_LINES, g_pdeWireVtxCnt);

    // 6 -- Axes + ticks
    drawBuffer(g_axesVbo, GL_LINES, g_axesVtxCnt);

    // 7 -- 2-D text labels
    drawLabels();
}

// -----------------------------------------------------------------------------
//  Entry point
// -----------------------------------------------------------------------------
int main() {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion = 1;
    attrs.minorVersion = 0;
    attrs.depth        = EM_TRUE;
    attrs.antialias    = EM_TRUE;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
        emscripten_webgl_create_context("#canvas", &attrs);
    emscripten_webgl_make_context_current(ctx);

    glEnable(GL_DEPTH_TEST);

    GLuint vs = compileShader(GL_VERTEX_SHADER,   VERT_SRC);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, FRAG_SRC);
    g_program = glCreateProgram();
    glAttachShader(g_program, vs);
    glAttachShader(g_program, fs);
    glLinkProgram(g_program);

    buildAxes();

    g_camera.distance  = 18.0f;
    g_camera.azimuth   = 0.6f;
    g_camera.elevation = 0.45f;

    emscripten_set_mousedown_callback("#canvas",
        nullptr, EM_FALSE, onMouseDown);
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT,
        nullptr, EM_FALSE, onMouseUp);
    emscripten_set_mousemove_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT,
        nullptr, EM_FALSE, onMouseMove);
    emscripten_set_wheel_callback("#canvas",
        nullptr, EM_FALSE, onWheel);

    emscripten_set_main_loop(mainLoop, 0, 1);
    return 0;
}
